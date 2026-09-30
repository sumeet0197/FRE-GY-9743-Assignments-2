from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import QuantLib as ql

from fixedincomelib.apis.date import qfCreateSchedule
from fixedincomelib.date import Date, Period, accrued, add_period
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention


__all__ = ["BondCalculator"]


class BondCalculator:
    """Fixed-rate bullet bond built from user-supplied terms, without case IDs.

    Supply a complete bond_convention dictionary, including issue date,
    first accrual date, first coupon date, maturity date and coupon rate:

        calculator = BondCalculator(bond_convention=bond_convention)
        cashflows = calculator.schedule_dataframe()

    """
    def __init__(self, issue_date: Optional[str] = None,
                 first_acc_date: Optional[str] = None,
                 first_cpn_date: Optional[str] = None,
                 maturity_date: Optional[str] = None,
                 coupon_rate: Optional[float] = None,
                 bond_convention: Optional[Dict[str, Any]] = None,
                 face_value: Optional[float] = None,
                 redemption: Optional[float] = None) -> None:
        self.conv = dict(bond_convention)
        # Bond terms must be inputs, never looked up in a Bloomberg case table.
        self.issue_date = issue_date if issue_date is not None else self.conv['issue date']
        self.first_acc_date = first_acc_date if first_acc_date is not None else self.conv['first accrual date']
        self.first_cpn_date = first_cpn_date if first_cpn_date is not None else self.conv['first coupon date']
        self.maturity_date = maturity_date if maturity_date is not None else self.conv['maturity date']
        self.coupon_rate = coupon_rate if coupon_rate is not None else self.conv['coupon rate']
        self.face_value = face_value if face_value is not None else self.conv['face value']
        self.redemption = redemption if redemption is not None else self.conv['redemption']
        self.coupon_period = Period(self.conv['coupon accrual period'])
        self.frequency = float(self.coupon_period.frequency())
        self.accrual_basis = AccrualBasis(self.conv['accrual basis'])
        self.first_accrual_basis = AccrualBasis(self.conv['first period accrual basis'])
        self.last_accrual_basis = AccrualBasis(self.conv['last period accrual basis'])
        self.schedule = self._generate_bond_schedule()
        # Dates first; each accrual method now owns its reference grid and coupon.
        for period in self.schedule:
            if period['period_type'] == 'first':
                period['coupon'] = self._first_period_accrual(period)
            elif period['period_type'] == 'last':
                period['coupon'] = self._last_period_accrual(period)
            else:
                period['coupon'] = self._regular_period_accrual(period)

    def settlement_date(self, value_date: str) -> Date:
        return add_period(
            Date(value_date), Period(self.conv['settlement offset']),
            BusinessDayConvention.new(self.conv['settlement business day convention']),
            HolidayConvention.new(self.conv['settlement holiday convention']))

    def yield_to_price(self, value_date: Optional[str] = None,
                       yield_rate: Optional[float] = None, *,
                       settlement_date: Optional[str] = None) -> Dict[str, Any]:
        """Discount remaining coupons and principal, then subtract accrued interest.

        1. Use Date(settlement_date) if supplied; otherwise call
           self.settlement_date(value_date). Keep periods with end_date > settle.
        2. Collect their coupon amounts. The first time increment is
           _periods_to_next_coupon(settle, first_period); each later increment
           is _periods_to_next_coupon(period['start_date'], period).
           Apply np.cumsum to get n, the time in coupon periods, not years.
        3. With decimal yield y and frequency f, use Street discount factors:
           multiple payments: b=1+y/f, DF=b**(-n);
           one payment: t=n/f, d=1+y*t, DF=1/d.
        4. Principal=face_value*redemption. Dirty=np.dot(coupons, DF)
           + principal*DF[-1]; AI=_accrued_interest(settle); clean=dirty-AI.
        5. Also return sensitivities to decimal y. For multiple payments:
           DF'=-n/f*b**(-n-1); DF''=n*(n+1)/f**2*b**(-n-2).
           For one payment: DF'=-t/d**2; DF''=2*t**2/d**3.
           Let discount_first=DF' and discount_second=DF''. Then:
           dpricedyield = np.dot(coupon_amounts, discount_first)
                         + redemption_amount * discount_first[-1]
           d2pricedyield2 = np.dot(coupon_amounts, discount_second)
                           + redemption_amount * discount_second[-1]
           These are the same cashflow sums as dirty price, with DF replaced
           by its first or second derivative. At fixed settlement, AI does
           not depend on yield, so clean and dirty prices have the same
           yield derivatives.
           Return clean_price, dirty_price, accrued_interest, dpricedyield,
           d2pricedyield2 and settlement_date (ISO string).

        ICMA + Street example: face 1000, 6%, quarterly, 8 coupons of 15,
        redemption 1.02, 70/90 of the first quarter remaining:
        n=[70/90, 70/90+1, ...]; at y=0.055, dirty=1030.4611,
        AI=3.3333, clean=1027.1278. Other accrual bases feed different
        accrual/time fractions into the same Street formulas above."""
        settle = Date(settlement_date) if settlement_date is not None else self.settlement_date(value_date)

        # A coupon whose accrual ends exactly at settlement has already been earned by the
        # seller, so it is not part of what the buyer pays for: the strict inequality is what
        # keeps yield_to_price and _accrued_interest looking at the same period.
        remaining = [p for p in self.schedule if p['end_date'] > settle]
        if not remaining:
            raise ValueError(f'No cashflows remain after {settle.ISO()}')

        coupon_amounts = np.array([p['coupon'] for p in remaining], dtype=float)

        # Time is measured in coupon periods, not years: the first increment runs from
        # settlement to the next coupon (a fraction of a period, or more than one across a
        # long stub), and every later increment is a one-period hop measured on that period's
        # own reference grid. cumsum turns the hops into times from settlement.
        increments = [self._periods_to_next_coupon(settle, remaining[0])]
        increments += [self._periods_to_next_coupon(p['start_date'], p) for p in remaining[1:]]
        n = np.cumsum(np.array(increments, dtype=float))

        y = float(yield_rate)
        f = self.frequency

        if len(remaining) > 1:
            # Street convention: compound at y/f over the fractional period count.
            base = 1.0 + y / f
            discount = base ** (-n)
            discount_first = -n / f * base ** (-n - 1.0)
            discount_second = n * (n + 1.0) / f ** 2 * base ** (-n - 2.0)
        else:
            # Street convention switches to simple interest once a single payment is left,
            # which is why a bond in its final period does not price as (1 + y/f)**(-n).
            t = n / f
            denominator = 1.0 + y * t
            discount = 1.0 / denominator
            discount_first = -t / denominator ** 2
            discount_second = 2.0 * t ** 2 / denominator ** 3

        redemption_amount = self.face_value * self.redemption

        dirty = np.dot(coupon_amounts, discount) + redemption_amount * discount[-1]

        # The same cashflow sums with the discount factor replaced by its derivatives. At a
        # fixed settlement date accrued interest carries no yield dependence, so these are the
        # derivatives of the clean price just as much as of the dirty price.
        first = np.dot(coupon_amounts, discount_first) + redemption_amount * discount_first[-1]
        second = np.dot(coupon_amounts, discount_second) + redemption_amount * discount_second[-1]

        ai = self._accrued_interest(settle)

        return {'clean_price': float(dirty - ai), 'dirty_price': float(dirty),
                'accrued_interest': ai, 'dpricedyield': float(first),
                'd2pricedyield2': float(second), 'settlement_date': settle.ISO()}

    def _generate_bond_schedule(self) -> List[Dict[str, Any]]:
        """Build dates first; the coupon methods will fill in interest amounts.

        1. Call qfCreateSchedule from first_acc_date to maturity_date using
           coupon period, schedule rule and end-of-month setting. Use 'NONE'
           for accrual basis and accrual-date business/holiday conventions;
           pass payment offset and payment conventions separately.
           Set first_regular_date=first_cpn_date (None if it equals maturity),
           and next_to_last_date=conv['last regular coupon date'].
           Payment keywords are payment_offset_business_day_convention and
           payment_offset_holiday_convention.
        2. Convert each row into a dict: start_date/end_date/payment_date
           as Date objects, is_regular as bool, plus period_type/accrual_basis.
           Label row 0 'first' with first_accrual_basis; otherwise the final
           row is 'last' with last_accrual_basis; others use 'regular' and
           accrual_basis. A single row is therefore 'first'.
        3. The API does not return IsRegular. Recreate the same ql.Schedule
           (unadjusted, with the same rule, tenor, anchors and end_of_month)
           to read isRegular(i+1). Also treat end=start+one coupon period as
           regular, using unadjusted add_period and the end-of-month setting.
           Save this bool as is_regular. Return the list; no coupons yet.

        Example (6M, BACKWARD, ISMA-30/360): KSA's final dates are
        2054-02-03 -> 2054-08-03 -> 2055-01-21, so the last period is short.
        Payment rolling may move payment_date, but never these accrual dates."""
        # first_regular_date anchors the end of an irregular first period. When the first
        # coupon date IS maturity there is only one period and nothing to anchor, and passing
        # it would make QuantLib reject the schedule.
        first_regular_date = self.first_cpn_date if self.first_cpn_date != self.maturity_date else None
        next_to_last_date = self.conv['last regular coupon date']

        # Accrual dates must be unadjusted: rolling them would change how much interest
        # accrues. Only payment dates are rolled, through the payment_* keywords. Accrual
        # basis is 'NONE' here because each period's own basis is applied later, by the
        # accrual methods, against the right reference grid.
        frame = qfCreateSchedule(
            start_date=self.first_acc_date,
            end_date=self.maturity_date,
            accrual_period=self.conv['coupon accrual period'],
            holiday_convention='NONE',
            business_day_convention='NONE',
            accrual_basis='NONE',
            rule=self.conv['schedule rule'],
            end_of_month=self.conv['end of month'],
            payment_offset=self.conv['payment offset'],
            payment_offset_business_day_convention=self.conv['payment business day convention'],
            payment_offset_holiday_convention=self.conv['payment holiday convention'],
            first_regular_date=first_regular_date,
            next_to_last_date=next_to_last_date,
        )

        # The public API returns dates but no IsRegular flag, so rebuild the identical
        # QuantLib schedule and read the flags off it. Same rule, tenor, anchors, calendar and
        # end-of-month setting, so the two schedules generate the same dates by construction.
        rule = ql.DateGeneration.Forward if str(self.conv['schedule rule']).upper() == 'FORWARD' \
            else ql.DateGeneration.Backward
        unadjusted = BusinessDayConvention.new('NONE')
        no_holidays = HolidayConvention.new('NONE')
        end_of_month = self.conv['end of month']
        ql_schedule = ql.Schedule(
            Date(self.first_acc_date),
            Date(self.maturity_date),
            self.coupon_period,
            no_holidays,
            unadjusted,
            unadjusted,
            rule,
            end_of_month,
            Date(first_regular_date),
            Date(next_to_last_date),
        )

        row_count = len(frame)
        periods = []
        for i, row in enumerate(frame.itertuples(index=False)):
            start_date = Date(row.StartDate)
            end_date = Date(row.EndDate)

            if i == 0:
                # Checked first, so a single-row schedule is 'first', never 'last'.
                period_type = 'first'
                accrual_basis = self.first_accrual_basis
            elif i == row_count - 1:
                period_type = 'last'
                accrual_basis = self.last_accrual_basis
            else:
                period_type = 'regular'
                accrual_basis = self.accrual_basis

            # isRegular is 1-indexed over periods. A period that spans exactly one coupon
            # period is regular too, even where QuantLib flags it otherwise at a schedule
            # boundary, so test that directly with the same unadjusted roll.
            natural_end = add_period(
                start_date, self.coupon_period, unadjusted, no_holidays, end_of_month)
            is_regular = bool(ql_schedule.isRegular(i + 1)) or end_date == natural_end

            periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'payment_date': Date(row.PaymentDate),
                'is_regular': is_regular,
                'period_type': period_type,
                'accrual_basis': accrual_basis,
            })

        return periods

    def _first_period_accrual(self, period: Dict[str, Any]) -> float:
        """Return the first coupon amount and save its accrual information.

        1. If is_regular, return _regular_period_accrual(period).
        2. Otherwise use period['accrual_basis']. Set ref_end=end_date and
           ref_start=ref_end minus one coupon period; save these initial
           dates in period['ref_start'] and period['ref_end'].
           Use add_period with BusinessDayConvention.new('NONE'),
           HolidayConvention.new('NONE') and end_of_month (native API values).
        3. If basis.needs_reference_period (ICMA), walk this grid BACKWARD
           until it covers start_date. For each overlap, add
           _year_fraction(max(start_date, ref_start), ref_end, basis,
                          ref_start, ref_end).
           Otherwise call _year_fraction(start_date, end_date, basis) once.
        4. Save the sum in period['accrual']; return face * rate * that sum.

        ICMA example: 8%, face 100, semiannual long first coupon:
        accrual=(58/181 + 1)/2; coupon=100*0.08*accrual=5.281768.
        Bond Basis 30/360 example: META's count is 191, so
        coupon=100*0.0525*(191/360)=2.785417. ACT/360 and ACT/365 FIXED
        use actual days/360 or /365 through the same _year_fraction helper."""
        if period['is_regular']:
            return self._regular_period_accrual(period)

        basis = period['accrual_basis']
        unadjusted = BusinessDayConvention.new('NONE')
        no_holidays = HolidayConvention.new('NONE')
        end_of_month = self.conv['end of month']

        # An irregular first period is anchored at its END: the quasi-coupon grid is built
        # backward from the first coupon date, because that is the date the regular schedule
        # would have hit. ref_start/ref_end saved here are the LAST quasi-period, the one the
        # bond is in immediately before its first coupon.
        ref_end = period['end_date']
        ref_start = add_period(
            ref_end, Period.negate_period(self.coupon_period), unadjusted, no_holidays, end_of_month)
        period['ref_start'] = ref_start
        period['ref_end'] = ref_end

        if basis.needs_reference_period:
            # ICMA measures each slice against the quasi-period it falls in, and a long stub
            # spans more than one such period with different day counts. Walk backward,
            # accumulating the overlap of each quasi-period with the real accrual period,
            # until the grid reaches back past start_date.
            full_accrual = 0.0
            slice_start, slice_end = ref_start, ref_end
            while True:
                overlap_start = max(period['start_date'], slice_start)
                full_accrual += self._year_fraction(
                    overlap_start, slice_end, basis, slice_start, slice_end)
                if slice_start <= period['start_date']:
                    break
                slice_end = slice_start
                slice_start = add_period(
                    slice_end, Period.negate_period(self.coupon_period),
                    unadjusted, no_holidays, end_of_month)
        else:
            # Every other basis has a fixed denominator, so the stub needs no splitting: one
            # day count across the whole period is the answer.
            full_accrual = self._year_fraction(period['start_date'], period['end_date'], basis)

        period['accrual'] = full_accrual
        return self.face_value * self.coupon_rate * full_accrual

    def _last_period_accrual(self, period: Dict[str, Any]) -> float:
        """Return the final coupon amount (no principal) and save its accrual.

        1. If is_regular, return _regular_period_accrual(period).
        2. Otherwise use period['accrual_basis']. Set ref_start=start_date
           and ref_end=ref_start plus one coupon period; save these initial
           dates in period['ref_start'] and period['ref_end'].
           Use add_period with BusinessDayConvention.new('NONE'),
           HolidayConvention.new('NONE') and end_of_month (native API values).
        3. If basis.needs_reference_period (ICMA), walk this grid FORWARD
           until it covers end_date. For each overlap, add
           _year_fraction(ref_start, min(end_date, ref_end), basis,
                          ref_start, ref_end).
           Otherwise call _year_fraction(start_date, end_date, basis) once.
        4. Save the sum in period['accrual']; return face * rate * that sum.

        ICMA example: 8%, face 100, semiannual; a 92-day short last period
        within a 184-day reference gives 100*0.08*(92/184/2)=2.
        ISMA-30/360 example: KSA, 2054-08-03 to 2055-01-21:
        coupon=100*0.0375*(168/360)=1.75. ACT/360 and ACT/365 FIXED
        use actual days/360 or /365 through the same _year_fraction helper."""
        if period['is_regular']:
            return self._regular_period_accrual(period)

        basis = period['accrual_basis']
        unadjusted = BusinessDayConvention.new('NONE')
        no_holidays = HolidayConvention.new('NONE')
        end_of_month = self.conv['end of month']

        # Mirror image of the first period: an irregular last period is anchored at its
        # START, since that is the last date the regular schedule did hit, and the grid runs
        # forward from there past maturity.
        ref_start = period['start_date']
        ref_end = add_period(
            ref_start, self.coupon_period, unadjusted, no_holidays, end_of_month)
        period['ref_start'] = ref_start
        period['ref_end'] = ref_end

        if basis.needs_reference_period:
            full_accrual = 0.0
            slice_start, slice_end = ref_start, ref_end
            while True:
                overlap_end = min(period['end_date'], slice_end)
                full_accrual += self._year_fraction(
                    slice_start, overlap_end, basis, slice_start, slice_end)
                if slice_end >= period['end_date']:
                    break
                slice_start = slice_end
                slice_end = add_period(
                    slice_start, self.coupon_period, unadjusted, no_holidays, end_of_month)
        else:
            full_accrual = self._year_fraction(period['start_date'], period['end_date'], basis)

        period['accrual'] = full_accrual
        return self.face_value * self.coupon_rate * full_accrual

    def _regular_period_accrual(self, period: Dict[str, Any]) -> float:
        """Save the regular period's year fraction and return its fixed coupon.

        1. Save start_date/end_date as period['ref_start']/period['ref_end'].
        2. Set period['accrual'] = _year_fraction(start_date, end_date,
           period['accrual_basis'], ref_start, ref_end).
        3. Return face_value * coupon_rate / frequency.

        ICMA example: face 1000, rate 6%, quarterly => accrual=0.25,
        coupon=1000*0.06/4=15. In this model, a regular coupon stays 15
        with other bases too; the saved year fraction changes with the basis
        and is used to calculate the earned share of that coupon."""
        # A regular period IS its own reference period: there is no stub to measure against
        # a neighbouring grid.
        period['ref_start'] = period['start_date']
        period['ref_end'] = period['end_date']
        period['accrual'] = self._year_fraction(
            period['start_date'], period['end_date'], period['accrual_basis'],
            period['ref_start'], period['ref_end'])

        # The coupon on a regular period is the contractual face * rate / frequency whatever
        # the basis says. The saved year fraction is what the basis changes, and it is only
        # used later to apportion this fixed coupon between seller and buyer.
        return self.face_value * self.coupon_rate / self.frequency

    def _accrued_interest(self, settlement_date: Date) -> float:
        """Return the earned share of the current coupon at settlement.

        1. Find the first schedule period with end_date > settlement_date.
           Equality moves to the next period, where accrued interest is zero.
        2. Calculate elapsed = _year_fraction(start_date, settlement_date,
           period['accrual_basis'], period['ref_start'], period['ref_end']).
           The helper/QuantLib handles the day count and ICMA reference grid;
           do not write another splitting loop in this method.
        3. Return period['coupon'] * elapsed / period['accrual'].

        ICMA example: coupon 15, 20 of 90 days elapsed => AI=15*20/90=3.333333.
        Bond Basis 30/360 example: META's first coupon is 2.785416667;
        96 of 191 counted days elapsed => AI=2.785416667*96/191=1.40.
        ACT/360 or ACT/365 FIXED uses actual-day year fractions in this ratio.
        The coupon already includes face and frequency; do not apply them again."""
        # Settlement exactly on an accrual end date belongs to the next period, where nothing
        # has accrued yet: the coupon ending that day is the seller's in full.
        period = next((p for p in self.schedule if p['end_date'] > settlement_date), None)
        if period is None:
            raise ValueError(f'Settlement date {settlement_date.ISO()} is past maturity')

        # _year_fraction hands ICMA its reference grid, and QuantLib does the quasi-period
        # decomposition internally, so no second splitting loop belongs here.
        ai_t = self._year_fraction(
            period['start_date'], settlement_date, period['accrual_basis'],
            period['ref_start'], period['ref_end'])
        full_period_length = period['accrual']

        # The ratio is what carries the basis: coupon already includes face and frequency, so
        # applying either again would double count.
        return period['coupon'] * ai_t / full_period_length

    def price_to_yield(self, value_date: Optional[str] = None, price: Optional[float] = None,
                       clean: bool = True, max_iter: int = 100, tol: float = 1e-12, *,
                       settlement_date: Optional[str] = None) -> Dict[str, Any]:
        """Provided solver: bracket the target price and bisect the yield interval.

        Assumes nonnegative coupons and a positive dirty price. The price falls
        as yield increases; keep the half-interval containing the target price.
        """
        settle = Date(settlement_date) if settlement_date is not None else self.settlement_date(value_date)
        accrued_interest = self._accrued_interest(settle)
        dirty_price = price + accrued_interest if clean else price
        lower = self._zcb_yield_guess(settle, dirty_price)
        upper = max(self.coupon_rate, lower + 0.05)
        while self.yield_to_price(settlement_date=settle.ISO(), yield_rate=upper)['dirty_price'] > dirty_price:
            upper = 2 * upper + self.frequency

        for _ in range(max_iter):
            ytm = (lower + upper) / 2
            result = self.yield_to_price(settlement_date=settle.ISO(), yield_rate=ytm)
            residual = result['dirty_price'] - dirty_price
            if abs(residual) <= tol:
                break
            if residual > 0:
                lower = ytm
            else:
                upper = ytm
        return {'yield': float(ytm), **result}

    def _zcb_yield_guess(self, settlement_date: Date, dirty_price: float) -> float:
        """Ignore coupons and solve the redemption-only Street pricing formula."""
        remaining = [p for p in self.schedule if p['end_date'] > settlement_date]
        n = self._periods_to_next_coupon(settlement_date, remaining[0])
        for period in remaining[1:]:
            n += self._periods_to_next_coupon(period['start_date'], period)
        redemption_amount = self.face_value * self.redemption
        if len(remaining) == 1:
            return self.frequency * (redemption_amount / dirty_price - 1) / n
        return self.frequency * ((redemption_amount / dirty_price)**(1 / n) - 1)

    def schedule_dataframe(self) -> pd.DataFrame:
        """Return generated period dates, accrual factors and cash amounts.

        Coupon excludes principal; Total includes redemption on the final row.
        Amounts use the supplied face value. No external data files are read.
        """
        rows = []
        for i, p in enumerate(self.schedule):
            principal = self.face_value * self.redemption if i == len(self.schedule) - 1 else 0.
            rows.append({'StartDate': p['start_date'].ISO(), 'EndDate': p['end_date'].ISO(),
                         'PaymentDate': p['payment_date'].ISO(),
                         'ReferenceStart': p['ref_start'].ISO(), 'ReferenceEnd': p['ref_end'].ISO(),
                         'IsRegular': p['is_regular'], 'Accrued': p['accrual'],
                         'Coupon': p['coupon'], 'Principal': principal, 'Total': p['coupon'] + principal})
        return pd.DataFrame(rows)

    def _periods_to_next_coupon(self, settlement_date, period):
        """Remaining time in coupon-period units, using the reference grid.

        Regular half-period remaining gives 0.5; a long stub can exceed 1.
        The denominator is a regular reference period, not the whole stub.
        """
        basis = period['accrual_basis']
        remaining = self._year_fraction(settlement_date, period['end_date'], basis,
                                        period['ref_start'], period['ref_end'])
        reference = self._year_fraction(period['ref_start'], period['ref_end'], basis,
                                        period['ref_start'], period['ref_end'])
        return remaining / reference

    def _year_fraction(self, start_date, end_date, basis, reference_period_start=None, reference_period_end=None):
        # ICMA needs reference dates; the new accrued API does not accept them.
        if basis.needs_reference_period:
            return basis.value.yearFraction(
                start_date, end_date, reference_period_start, reference_period_end)
        return accrued(start_date, end_date, basis.value,
                       BusinessDayConvention.new('NONE'), HolidayConvention.new('NONE'))
