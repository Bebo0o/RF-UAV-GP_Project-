import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time

# =========================
# Receivers positions
# =========================
receivers = np.array([[0,0],[100,0],[50,100]])
drone_types = ['DJI FPV','DJI Mavic','DEVENTION DEVO','DAUTEL EVO']

# =========================
# Signals
# =========================
def generate_fm_signal(length=500):
    t = np.linspace(0,1,length)
    f0, f1 = 5, 20
    return np.sin(2*np.pi*(f0 + f1*np.sin(2*np.pi*0.2*t))*t)

def receive_signal(drone_signal,distance):
    noise = np.random.normal(0,0.1,size=drone_signal.shape)
    attenuation = np.exp(-distance/50)
    return drone_signal*attenuation + noise

# =========================
# Trilateration (estimate position)
# =========================
def rssi_to_distance(rssi_val):
    # simple conversion for demo
    return np.exp(-rssi_val/50)

def trilateration(rx_positions, rx_distances):
    # naive approximation for demo purposes
    x = np.mean([p[0] for p in rx_positions])
    y = np.mean([p[1] for p in rx_positions])
    return np.array([x,y])

# =========================
# GUI
# =========================
class RFTrackerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Drone Tracker")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Dropdown to select drone type
        self.combo = QComboBox()
        self.combo.addItems(drone_types)
        self.layout.addWidget(QLabel("Select Drone Type:"))
        self.layout.addWidget(self.combo)

        # Start button
        self.btn = QPushButton("Start Simulation")
        self.btn.clicked.connect(self.start_simulation)
        self.layout.addWidget(self.btn)

        # Drone info label
        self.label = QLabel("Drone Info: ")
        self.layout.addWidget(self.label)

        # Matplotlib Figure
        self.figure = Figure(figsize=(8,8))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.axes = [self.figure.add_subplot(len(receivers)+1,1,i+1) for i in range(len(receivers)+1)]

    def start_simulation(self):
        selected_type = self.combo.currentText()
        for t in range(20):
            # Drone position and signal
            drone_pos = np.random.randint(10,90,2)
            drone_signal = generate_fm_signal()
            self.label.setText(f"Drone Type: {selected_type}  Pos: {drone_pos}")

            # Drone signal subplot
            self.axes[0].clear()
            self.axes[0].plot(drone_signal, color='green')
            self.axes[0].set_title("Drone Emitted Signal")

            # Receiver signals
            rx_distances = []
            for i, (rx_x, rx_y) in enumerate(receivers):
                self.axes[i+1].clear()
                distance = np.linalg.norm(drone_pos - np.array([rx_x, rx_y]))
                rx_distances.append(distance)
                rx_signal = receive_signal(drone_signal, distance)
                self.axes[i+1].plot(rx_signal, color='blue')
                self.axes[i+1].set_title(f"Receiver {i+1} Signal (Dist: {distance:.1f})")

            # Estimate position (simple mean for demo)
            predicted_pos = trilateration(receivers, rx_distances)

            # Plot positions
            for ax in self.axes:
                ax.set_xlim(0,500)
                ax.set_ylim(-1.5,1.5)
            self.canvas.draw()
            QApplication.processEvents()
            time.sleep(0.5)

# =========================
# Run App
# =========================
app = QApplication(sys.argv)
window = RFTrackerGUI()
window.show()
sys.exit(app.exec_())