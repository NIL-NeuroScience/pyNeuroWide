# %%
from pyNeuroWide import io, utils
from pyNeuroWide import process as pnw
import numpy as np
import json
import os
import argparse
from pyNeuroWide.allenAtlas import AlignWindow
from PyQt5.QtWidgets import QApplication
import sys

# %%

app = QApplication(sys.argv)
ref = np.random.rand(256, 256)
win = AlignWindow(ref)
win.show()
sys.exit(app.exec_())