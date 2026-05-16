import os.path
import pathlib

import matplotlib
from எண்ணிக்கை import உரைக்கு as உ

स्रोत = pathlib.Path(__file__).parent.resolve()
परिणाम_नत्थी = os.path.join(स्रोत, "परिणाम")

# परिणान का नत्थी बनाना अगर अब तक नहीं बना गया है
if not os.path.exists(परिणाम_नत्थी):
    os.makedirs(परिणाम_नत्थी)

from matplotlib import rcParams
rcParams['font.family'] = 'Noto Sans'

def formatter(axe):
        axe.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: உ(int(x), 'देवनागरी'))
        )
        axe.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: உ(int(x), 'देवनागरी'))
        )
