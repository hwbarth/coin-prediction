import talib
#OVERLAP STUDIES:

def calculate_bollinger_bands(data, time_period=20):
    close = data[:, 4]  # Close prices
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], upperband, middleband, lowerband))

def calculate_dema(data, time_period=30):
    close = data[:, 4]  # Close prices
    dema = talib.DEMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], dema))

def calculate_ema(data, time_period=30):
    close = data[:, 4]  # Close prices
    ema = talib.EMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], ema))

def calculate_hilbert_trendline(data):
    close = data[:, 4]  # Close prices
    h_trendline = talib.HT_TRENDLINE(close)
    return np.column_stack((data[:, 0], h_trendline))

def calculate_kamam(data, time_period=30):
    close = data[:, 4]  # Close prices
    kmam = talib.KAMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], kmam))

def calculate_midpoint(data, time_period=14):
    close = data[:, 4]  # Close prices
    midpoint = talib.MIDPOINT(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], midpoint))

def calculate_midprice(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]  # Low prices
    midprice = talib.MIDPRICE(high, low, timeperiod=time_period)
    return np.column_stack((data[:, 0], midprice))

def calculate_sar(data, acceleration=0.02, maximum=0.2):
    high = data[:, 2]  # High prices
    low = data[:, 3]  # Low prices
    sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    return np.column_stack((data[:, 0], sar))

def calculate_sma(data, time_period=30):
    close = data[:, 4]  # Close prices
    sma = talib.SMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], sma))

def calculate_tema(data, time_period=30):
    close = data[:, 4]  # Close prices
    tema = talib.TEMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], tema))

def calculate_wma(data, time_period=30):
    close = data[:, 4]  # Close prices
    wma = talib.WMA(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], wma))


#momentum indicators:
def calculate_adx(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    adx = talib.ADX(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], adx))

def calculate_adxr(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    adxr = talib.ADXR(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], adxr))

def calculate_apO(data):
    close = data[:, 4] # Close prices
    apo = talib.APO(close)
    return np.column_stack((data[:, 0], apo))

def calculate_aroonosc(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    aroonosc = talib.AROONOSC(high, low)
    return np.column_stack((data[:, 0], aroonosc))

def calculate_bop(data):
    open_price = data[:, 1]  # Open prices
    high = data[:, 2]        # High prices
    low = data[:, 3]         # Low prices
    close = data[:, 4]       # Close prices
    bop = talib.BOP(open_price, high, low, close)
    return np.column_stack((data[:, 0], bop))

def calculate_cci(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    cci = talib.CCI(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], cci))

def calculate_cmo(data, time_period=14):
    close = data[:, 4]  # Close prices
    cmo = talib.CMO(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], cmo))

def calculate_dx(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    dx = talib.DX(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], dx))

def calculate_macd(data):
    close = data[:, 4]  # Close prices
    macd, macdsignal, macdhist = talib.MACD(close)
    return np.column_stack((data[:, 0], macd, macdsignal, macdhist))

def calculate_minus_di(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    minus_di = talib.MINUS_DI(high, low, close)
    return np.column_stack((data[:, 0], minus_di))

def calculate_minus_dm(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    minus_dm = talib.MINUS_DM(high, low)
    return np.column_stack((data[:, 0], minus_dm))

def calculate_momentum(data, time_period=10):
    close = data[:, 4]  # Close prices
    momentum = talib.MOM(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], momentum))

def calculate_plus_di(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    plus_di = talib.PLUS_DI(high, low, close)
    return np.column_stack((data[:, 0], plus_di))

def calculate_plus_dm(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    plus_dm = talib.PLUS_DM(high, low)
    return np.column_stack((data[:, 0], plus_dm))

def calculate_ppo(data):
    close = data[:, 4]  # Close prices
    ppo = talib.PPO(close)
    return np.column_stack((data[:, 0], ppo))

def calculate_roc(data, time_period=10):
    close = data[:, 4]  # Close prices
    roc = talib.ROC(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], roc))

def calculate_rocp(data):
    close = data[:, 4]  # Close prices
    rocp = talib.ROCP(close)
    return np.column_stack((data[:, 0], rocp))

def calculate_rocr(data):
    close = data[:, 4]  # Close prices
    rocr = talib.ROCR(close)
    return np.column_stack((data[:, 0], rocr))

def calculate_rocr100(data):
    close = data[:, 4]  # Close prices
    rocr100 = talib.ROCR100(close)
    return np.column_stack((data[:, 0], rocr100))

def calculate_rsi(data, time_period=14):
    close = data[:, 4]  # Close prices
    rsi = talib.RSI(close, timeperiod=time_period)
    return np.column_stack((data[:, 0], rsi))

def calculate_stochastic(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    slowk, slowd = talib.STOCHF(high, low, close)
    return np.column_stack((data[:, 0], slowk, slowd))

def calculate_trix(data):
    close = data[:, 4]  # Close prices
    trix = talib.TRIX(close)
    return np.column_stack((data[:, 0], trix))

def calculate_ultosc(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    ultosc = talib.ULTOSC(high, low, close)
    return np.column_stack((data[:, 0], ultosc))

def calculate_willr(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    willr = talib.WILLR(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], willr))

#volatility indicators:

def calculate_atr(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    atr = talib.ATR(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], atr))

def calculate_natr(data, time_period=14):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    close = data[:, 4] # Close prices
    natr = talib.NATR(high, low, close, timeperiod=time_period)
    return np.column_stack((data[:, 0], natr))

def calculate_trange(data):
    high = data[:, 2]  # High prices
    low = data[:, 3]   # Low prices
    prev_close = np.roll(data[:, 4], 1)  # Previous close prices (shifted)
    prev_close[0] = low[0]  # Handle the first element
    true_range = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.column_stack((data[:, 0], true_range))

# Example usage:
# data = np.array([[timestamp, open, high, low, close, volume], ...])
# atr_result = calculate_atr(data)
# natr_result = calculate_natr(data)
# trange_result = calculate_trange(data)

# PATTERN RECOGNITION:
import numpy as np
# import talib

def calculate_two_crows(data):
    two_crows = talib.CDL2CROWS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], two_crows))

def calculate_three_black_crows(data):
    three_black_crows = talib.CDL3BLACKCROWS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_black_crows))

def calculate_three_inside(data):
    three_inside = talib.CDL3INSIDE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_inside))

def calculate_three_line_strike(data):
    three_line_strike = talib.CDL3LINESTRIKE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_line_strike))

def calculate_three_outside(data):
    three_outside = talib.CDL3OUTSIDE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_outside))

def calculate_three_stars_in_south(data):
    three_stars_in_south = talib.CDL3STARSINSOUTH(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_stars_in_south))

def calculate_three_advancing_white(data):
    three_advancing_white = talib.CDL3WHITESOLDIERS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], three_advancing_white))

def calculate_abandoned_baby(data):
    abandoned_baby = talib.DLABANDONEDBABY(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], abandoned_baby))

def calculate_advance_block(data):
    advance_block = talib.CDLADVANCEBLOCK(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], advance_block))

def calculate_belt_hold(data):
    belt_hold = talib.CDLBELTHOLD(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], belt_hold))

def calculate_breakaway(data):
    breakaway = talib.CDLBREAKAWAY(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], breakaway))

def calculate_closing_marubozu(data):
    closing_marubozu = talib.CDLCLOSINGMARUBOZU(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], closing_marubozu))

def calculate_concealing_baby_swallow(data):
    concealing_baby_swallow = talib.DLCONCEALBABYSWALL(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], concealing_baby_swallow))

def calculate_counterattack(data):
    counterattack = talib.CDLCOUNTERATTACK(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], counterattack))

def calculate_dark_cloud_cover(data):
    dark_cloud_cover = talib.CLDARKCLOUDCOVER(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], dark_cloud_cover))

def calculate_doji(data):
    doji = talib.DLDOJI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], doji))

def calculate_doji_star(data):
    doji_star = talib.CDLDOJISTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], doji_star))

def calculate_dragonfly_doji(data):
    dragonfly_doji = talib.DLDRAGONFLYDOJI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], dragonfly_doji))

def calculate_engulfing(data):
    engulfing = talib.DLENGULFING(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], engulfing))

def calculate_evening_doji_star(data):
    evening_doji_star = talib.CDLEVENINGDOJISTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], evening_doji_star))

def calculate_evening_star(data):
    evening_star = talib.CDLEVENINGSTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], evening_star))

def calculate_up_down_gap_side(data):
    up_down_gap_side = talib.CDLGAPSIDESIDEWHITE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], up_down_gap_side))

def calculate_gravestone_doji(data):
    gravestone_doji = talib.CDLGRAVESTONEDOJI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], gravestone_doji))

def calculate_hammer(data):
    hammer = talib.CDLHAMMER(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], hammer))

def calculate_hanging_man(data):
    hanging_man = talib.CDLHAMMER(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], hanging_man))

def calculate_harami(data):
    harami = talib.CDLHARAMI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], harami))

def calculate_harami_cross(data):
    harami_cross = talib.DLHARAMICROSS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], harami_cross))

def calculate_high_wave(data):
    high_wave = talib.CDLHIGHWAVE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], high_wave))

def calculate_hikkake(data):
    hikkake = talib.CDLHIKKAKE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], hikkake))

def calculate_modified_hikkake(data):
    modified_hikkake = talib.CDLHIKKAKEMOD(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], modified_hikkake))

def calculate_homing_pigeon(data):
    homing_pigeon = talib.CDLHOMINGPIGEON(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], homing_pigeon))

def calculate_identical_three_crows(data):
    identical_three_crows = talib.CDLIDENTICAL3CROWS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], identical_three_crows))

def calculate_in_neck(data):
    in_neck = talib.CDLINNECK(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], in_neck))

def calculate_inverted_hammer(data):
    inverted_hammer = talib.DLINVERTEDHAMMER(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], inverted_hammer))

def calculate_kicking(data):
    kicking = talib.DLKICKING(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], kicking))

def calculate_kicking_by_length(data):
    kicking_by_length = talib.CDLKICKINGBYLENGTH(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], kicking_by_length))

def calculate_ladder_bottom(data):
    ladder_bottom = talib.CDLLADDERBOTTOM(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], ladder_bottom))

def calculate_long_legged_doji(data):
    long_legged_doji = talib.CDLLONGLEGGEDDOJI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], long_legged_doji))

def calculate_long_line(data):
    long_line = talib.CDLLONGLINE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], long_line))

def calculate_marubozu(data):
    marubozu = talib.CDLMARUBOZU(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], marubozu))

def calculate_matching_low(data):
    matching_low = talib.CDLMATCHINGLOW(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], matching_low))

def calculate_mat_hold(data):
    mat_hold = talib.CDLMATHOLD(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], mat_hold))

def calculate_morning_doji_star(data):
    morning_doji_star = talib.CDLMORNINGDOJISTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], morning_doji_star))

def calculate_morning_star(data):
    morning_star = talib.CDLMORNINGSTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], morning_star))

def calculate_on_neck(data):
    on_neck = talib.CDLONNECK(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], on_neck))

def calculate_piercing(data):
    piercing = talib.CDLPIERCING(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], piercing))

def calculate_rickshaw_man(data):
    rickshaw_man = talib.DLRICKSHAWMAN(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], rickshaw_man))

def calculate_rising_falling_three_methods(data):
    rising_falling_three_methods = talib.CDLRISEFALL3METHODS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], rising_falling_three_methods))

def calculate_separating_lines(data):
    separating_lines = talib.CDLSEPARATINGLINES(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], separating_lines))

def calculate_shooting_star(data):
    shooting_star = talib.CDLSHOOTINGSTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], shooting_star))

def calculate_short_line(data):
    short_line = talib.CDLSHORTLINE(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], short_line))

def calculate_spinning_top(data):
    spinning_top = talib.CDLSPINNINGTOP(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], spinning_top))

def calculate_stalled_pattern(data):
    stalled_pattern = talib.CDLSTALLEDPATTERN(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], stalled_pattern))

def calculate_stick_sandwich(data):
    stick_sandwich = talib.CDLSTICKSANDWICH(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], stick_sandwich))

def calculate_takuri(data):
    takuri = talib.CDLTAKURI(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], takuri))

def calculate_tasuki_gap(data):
    tasuki_gap = talib.CDLTASUKIGAP(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], tasuki_gap))

def calculate_thrusting(data):
    thrusting = talib.CDLTHRUSTING(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], thrusting))

def calculate_tristar(data):
    tristar = talib.DLTRISTAR(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], tristar))

def calculate_unique_three_river(data):
    unique_three_river = talib.CDLUNIQUE3RIVER(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], unique_three_river))

def calculate_upside_gap_two_crows(data):
    upside_gap_two_crows = talib.CDLUPSIDEGAP2CROWS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], upside_gap_two_crows))

def calculate_upside_downside_gap_three_methods(data):
    upside_downside_gap_three_methods = talib.CDLXSIDEGAP3METHODS(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    return np.column_stack((data[:, 0], upside_downside_gap_three_methods))

# Example usage:
# data = np.array([[timestamp, open, high, low, close, volume], ...])
# two_crows_result = calculate_two_crows(data)
# three_black_crows_result = calculate_three_black_crows(data)

#CYCLE INDICATORS:
import numpy as np


def calculate_hilbert_transform_dominant_cycle_period(data):
    """
    Calculates the Hilbert Transform - Dominant Cycle Period.
    
    Parameters:
        data (np.ndarray): A 6D array with the following fields:
                           [timestamp, open, high, low, close, volume]
    
    Returns:
        np.ndarray: A NumPy array with the timestamp and the calculated cycle period.
    """
    dcp = talib.HT_DCPERIOD(data[:, 4])  # Using close prices for calculation
    return np.column_stack((data[:, 0], dcp))

def calculate_hilbert_transform_dominant_cycle_phase(data):
    """
    Calculates the Hilbert Transform - Dominant Cycle Phase.
    
    Parameters:
        data (np.ndarray): A 6D array with the following fields:
                           [timestamp, open, high, low, close, volume]
    
    Returns:
        np.ndarray: A NumPy array with the timestamp and the calculated cycle phase.
    """
    dph = talib.HT_DCPHASE(data[:, 4])  # Using close prices for calculation
    return np.column_stack((data[:, 0], dph))

def calculate_hilbert_transform_trend_mode(data):
    """
    Calculates the Hilbert Transform - Trend vs Cycle Mode.
    
    Parameters:
        data (np.ndarray): A 6D array with the following fields:
                           [timestamp, open, high, low, close, volume]
    
    Returns:
        np.ndarray: A NumPy array with the timestamp and the calculated trend mode.
    """
    trend_mode = talib.HT_TRENDMODE(data[:, 4])  # Using close prices for calculation
    return np.column_stack((data[:, 0], trend_mode))




