import numpy as np
import sys
import pandas as pd
import ast
import importlib.resources as pkg_resources
from importlib.resources import files
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QDialog, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import cv2
from matplotlib.figure import Figure

def loadAllenRegions():
    data_path = files("pyNeuroWide.data") / "allenRegions.csv"

    with data_path.open('r') as f:
        df = pd.read_csv(f)
    df['x_L'] = df['x_L'].apply(ast.literal_eval)
    df['y_L'] = df['y_L'].apply(ast.literal_eval)
    df['x_R'] = df['x_R'].apply(ast.literal_eval)
    df['y_R'] = df['y_R'].apply(ast.literal_eval)

    regions = list(zip(df['x_L'],df['y_L'],df['x_R'],df['y_R']))

    return df,regions

def coords_to_mask(transformed_coords, image_shape):
    """
    transformed_coords: (N, 2) array of XY points
    image_shape: (H, W)
    """
    # Convert to integer pixel coordinates
    polygon = np.round(transformed_coords.T).astype(np.int32)

    # Ensure it's shaped (1, N, 2) for fillPoly
    polygon = polygon[np.newaxis, :, :]

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 1)

    return mask.astype(bool)

def transform_coords(coords, scale, angle_deg, tx, ty):
    # Step 1: Compute centroid
    regions = coords
    all_x_L = np.concatenate([np.array(x_L) for x_L, y_L, x_R, y_R in regions])
    all_y_L = np.concatenate([np.array(y_L) for x_L, y_L, x_R, y_R in regions])
    all_x_R = np.concatenate([np.array(x_R) for x_L, y_L, x_R, y_R in regions])
    all_y_R = np.concatenate([np.array(y_R) for x_L, y_L, x_R, y_R in regions])

    center_x = np.mean(np.concatenate([all_x_L,all_x_R]))
    center_y = np.mean(np.concatenate([all_y_L,all_y_R]))
    
    # Step 2: Center the coordinates
    centered = [
        (np.array(x_L) - center_x, np.array(y_L) - center_y, np.array(x_R) - center_x, np.array(y_R) - center_y)
        for x_L, y_L, x_R, y_R in regions
    ]
    
    # Step 3: Apply rotation and scale
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    def apply_linear_transform(x_L, y_L, x_R, y_R, R, scale):
        coords_L = np.vstack([x_L, y_L])  # shape (2, N)
        new_coords_L = scale * R @ coords_L    # shape (2, N)

        coords_R = np.vstack([x_R, y_R])  # shape (2, N)
        new_coords_R = scale * R @ coords_R  # shape (2, N)

        return new_coords_L[0], new_coords_L[1], new_coords_R[0], new_coords_R[1]

    # Apply to each region
    transformed_regions = [
        apply_linear_transform(np.array(x_L), np.array(y_L), np.array(x_R), np.array(y_R), R, scale)
        for x_L, y_L, x_R, y_R in centered
    ]

    # Step 4: Translate back to center and apply user shift
    transformed = [
        (np.array(x_L) + center_x + tx, np.array(y_L) + center_y + ty, np.array(x_R) + center_x + tx, np.array(y_R) + center_y + ty)
        for x_L, y_L, x_R, y_R in transformed_regions
    ]
    return transformed

class AlignWindow(QDialog):
    def __init__(self, reference_img):
        super().__init__()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setWindowTitle("Allen Atlas Registration")

        self.df,self.mask_coords = loadAllenRegions()

        # self.df = df
        self.ref_img = reference_img
        # self.mask_coords = mask_coords

        all_x_L = np.concatenate([np.array(x_L) for x_L, y_L, x_R, y_R in self.mask_coords])
        all_y_L = np.concatenate([np.array(y_L) for x_L, y_L, x_R, y_R in self.mask_coords])
        all_x_R = np.concatenate([np.array(x_R) for x_L, y_L, x_R, y_R in self.mask_coords])
        all_y_R = np.concatenate([np.array(y_R) for x_L, y_L, x_R, y_R in self.mask_coords])

        center_x = np.mean(np.concatenate([all_x_L,all_x_R]))
        center_y = np.mean(np.concatenate([all_y_L,all_y_R]))

        ref_center_x = self.ref_img.shape[1] // 2
        ref_center_y = self.ref_img.shape[0] // 2

        # Initial transform params
        self.scale = 1.4
        self.angle = 0.0
        self.tx = np.round(ref_center_x - center_x)
        self.ty = np.round(ref_center_y - center_y) + 40

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_alpha(0.0)
        self.ax.set_facecolor('none')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;")
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.canvas.setAutoFillBackground(False)

        self.im_clim = np.percentile(self.ref_img, [1, 99])

        self.im = self.ax.imshow(self.ref_img, cmap='viridis')
        self.im.set_clim(self.im_clim[0], self.im_clim[1])
        self.cbar = self.fig.colorbar(self.im,ax=self.ax)
        self.cbar.ax.tick_params(colors=[0.9,0.9,0.9])

        # add instructions
        self.instructions = QLabel("R/T - rotate mask\n+/- - scale mask\nArrow keys - shift mask\nEnter - finish\nEscape - cancel")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: #cccccc")

        # add clim controls
        self.lim_ctrl = QHBoxLayout()
        self.lower_label = QLabel('Lower CLim:')
        self.lower_clim = QLineEdit()
        self.upper_label = QLabel('Upper CLim:')
        self.upper_clim = QLineEdit()
        self.confirm_clim = QPushButton("Set CLim")
        self.confirm_clim.clicked.connect(self.set_clim)

        self.lower_clim.setText(f"{self.im_clim[0]:.2f}")
        self.upper_clim.setText(f"{self.im_clim[1]:.2f}")

        self.lim_ctrl.addWidget(self.lower_label)
        self.lim_ctrl.addWidget(self.lower_clim)
        self.lim_ctrl.addWidget(self.upper_label)
        self.lim_ctrl.addWidget(self.upper_clim)
        self.lim_ctrl.addWidget(self.confirm_clim)

        # add everything to layout
        layout = QVBoxLayout()
        layout.addWidget(self.instructions)
        layout.addLayout(self.lim_ctrl)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        # container = QWidget()
        # container.setLayout(layout)
        # self.setCentralWidget(container)

        self.draw_overlay()
        self.setFocus()

    def set_clim(self):
        try:
            lower = float(self.lower_clim.text())
            upper = float(self.upper_clim.text())
            if lower < upper:
                self.im_clim = (lower, upper)
                self.draw_overlay()
            else:
                print("Invalid CLim range")
        except ValueError:
            print("Invalid input for CLim")
        
        # self.confirm_clim.setDown(False)
        # self.confirm_clim.repaint()
        self.lower_clim.clearFocus()
        self.upper_clim.clearFocus()
        self.setFocus()

    def draw_overlay(self):
        self.ax.clear()
        self.im = self.ax.imshow(self.ref_img, cmap='viridis')
        self.im.set_clim(self.im_clim[0], self.im_clim[1])
        transformed = transform_coords(self.mask_coords, self.scale, self.angle, self.tx, self.ty)
        
        for x_L,y_L,x_R,y_R in transformed:
            self.ax.plot(x_L, y_L, color='red', linewidth=2, alpha=0.5)
            self.ax.plot(x_R, y_R, color='red', linewidth=2, alpha=0.5)
        
        self.ax.axis('off')
        self.ax.set_title(f"Scale={self.scale:.2f}, Angle={self.angle:.1f}, Shift=({self.tx},{self.ty})",color=[0.9,0.9,0.9])
        self.cbar.update_normal(self.im)
        self.canvas.draw()

    def keyPressEvent(self, event):
        step = 1
        if event.key() == Qt.Key_Left:
            self.tx -= step
        elif event.key() == Qt.Key_Right:
            self.tx += step
        elif event.key() == Qt.Key_Up:
            self.ty -= step
        elif event.key() == Qt.Key_Down:
            self.ty += step
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.scale *= 1.01
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            self.scale /= 1.01
        elif event.key() == Qt.Key_R:
            self.angle += 0.5
        elif event.key() == Qt.Key_T:
            self.angle -= 0.5
        elif event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Return:
            if self.lower_clim.hasFocus() or self.upper_clim.hasFocus():
                self.set_clim()
            else:
                self.finish()

        self.draw_overlay()

    def get_pixel_mask(self,mask_coords):
        transformed = transform_coords(self.mask_coords, self.scale, self.angle, self.tx, self.ty)
        return coords_to_mask(transformed, self.ref_img.shape)
    
    def finish(self):
        transformed = transform_coords(self.mask_coords, self.scale, self.angle, self.tx, self.ty)
        data = []
        regionNames = self.df['region_short']
        regionNames_full = self.df['region_full']

        for i,name in enumerate(regionNames):
            L_coords = np.array(transformed[i][0:2])
            mask_L = coords_to_mask(L_coords, self.ref_img.shape)
            R_coords = np.array(transformed[i][2:4])
            mask_R = coords_to_mask(R_coords, self.ref_img.shape)

            data.append({
                'region_idx': i,
                'region_short': name,
                'region_full': regionNames_full[i],
                'mask_L': mask_L,
                'mask_R': mask_R
            })

        self.masks_df = pd.DataFrame(data)
        self.accept()  # closes the window

# To test manually
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ref = np.random.rand(256, 256)
    win = AlignWindow(ref)
    win.show()
    sys.exit(app.exec_())
