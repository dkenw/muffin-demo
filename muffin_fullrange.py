import numpy as np


class Pool:
    size:               int
    liquidity_arr:      np.ndarray
    sqrt_p_arr:         np.ndarray
    sqrt_gamma_arr:     np.ndarray
    fee0_growth_arr:    np.ndarray
    fee1_growth_arr:    np.ndarray

    def __init__(self, liquidity_arr=[10000, 10000], sqrt_gamma_arr=[0.9985, 0.9997], sqrt_p=1.0):
        self.size = len(liquidity_arr)
        self.sqrt_gamma_arr = np.array(sqrt_gamma_arr, dtype=np.float_)
        self.liquidity_arr = np.array(liquidity_arr, dtype=np.float_)
        self.sqrt_p_arr = np.array([sqrt_p] * len(liquidity_arr), dtype=np.float_)
        self.fee0_growth_arr = np.array([0.] * self.size)
        self.fee1_growth_arr = np.array([0.] * self.size)

    @property
    def prices(self):
        return self.sqrt_p_arr ** 2

    @property
    def price(self):
        """
        a combined price weighted by liquidity
        """
        return np.sum(self.prices * self.liquidity_arr) / np.sum(self.liquidity_arr)

    def swap(self, is_token0: bool, amt_desired: float):
        # in this prototype, we only support "exact_input" order (i.e. has specified amount to sell)
        is_exact_in = amt_desired > 0
        assert is_exact_in

        # calculate input amt of each tier
        amts_in, mask = self._calc_tier_amts_in(is_token0, amt_desired)

        # calculate input amt of each tier after charging fee
        gamma = self.sqrt_gamma_arr ** 2
        amts_in_after_fee = np.zeros(self.size)
        amts_in_after_fee[mask] = amts_in[mask] * gamma[mask]

        # calculate new sqrt price of each tier after swap
        sqrt_p_new = np.zeros(self.size)
        sqrt_p_new[mask] = calc_sqrt_p_from_amt(is_token0, self.sqrt_p_arr[mask], self.liquidity_arr[mask], amts_in_after_fee[mask])  # nopep8

        # calculate amt out of each tier
        amts_out = np.zeros(self.size)
        amts_out[mask] = calc_amt_from_sqrt_p(not is_token0, self.sqrt_p_arr[mask], sqrt_p_new[mask], self.liquidity_arr[mask])  # nopep8

        # calculate fee amts
        fee_amts = np.zeros(self.size)
        fee_amts[mask] = amts_in[mask] - amts_in_after_fee[mask]

        # update sqrt price state
        self.sqrt_p_arr[mask] = sqrt_p_new[mask]

        # update fee growth state
        fee_growth_arr = self.fee0_growth_arr if is_token0 == is_exact_in else self.fee1_growth_arr
        fee_growth_arr[mask] += fee_amts[mask] / self.liquidity_arr[mask]

        return {
            'amt_in': np.sum(amts_in[mask]),
            'amt_out': np.sum(amts_out[mask]),
            'fee_amt': np.sum(fee_amts[mask]),
            'fee_bps': np.sum(fee_amts[mask]) / np.sum(amts_in[mask]) * 10000,
            'amts_in': amts_in,
            'amts_out': amts_out,
            'fee_amts': fee_amts,
        }

    def _calc_tier_amts_in(self, is_token0: bool, amount: float):
        assert amount >= 0, 'for exact-input order only'

        # gamma: array of (1 - percentage fee)
        # lsg: array of liquidity divided by sqrt_gamma
        # res: array of token reserve divided by gamma
        gamma = self.sqrt_gamma_arr ** 2
        lsg = self.liquidity_arr / self.sqrt_gamma_arr
        res = ((self.liquidity_arr / self.sqrt_p_arr) / gamma if is_token0 else
               (self.liquidity_arr * self.sqrt_p_arr) / gamma)

        # mask: array of boolean of whether the tier will be used
        # amts: array of input amount routed to each tier
        mask = np.full(self.size, True)
        amts = np.zeros(self.size)

        # calculate amts with lagrange multiplier method, then reject those tiers with negative input amts.
        # repeat until all input amts are non-negative
        while True:
            amts[mask] = (lsg[mask] * (amount + np.sum(res[mask])) / np.sum(lsg[mask])) - res[mask]
            if np.all(amts[mask] >= 0):
                amts[~mask] = 0
                break
            mask &= amts >= 0
        return amts, mask


def calc_sqrt_p_from_amt(is_token0: bool, sqrt_p0, liquidity, amt):
    if is_token0:
        # √P1 = L √P0 / (L + √P0 * Δx)
        return (liquidity * sqrt_p0) / (liquidity + (amt * sqrt_p0))
    else:
        # √P1 = √P0 + (Δy / L)
        return sqrt_p0 + (amt / liquidity)


def calc_amt_from_sqrt_p(is_token0: bool, sqrt_p0, sqrt_p1, liquidity):
    if is_token0:
        # amt0 = L * (√P0 - √P1) / √P0√P1
        return liquidity * (sqrt_p0 - sqrt_p1) / (sqrt_p0 * sqrt_p1)
    else:
        # amt1 = L * (√P0 - √P1)
        return liquidity * (sqrt_p1 - sqrt_p0)
