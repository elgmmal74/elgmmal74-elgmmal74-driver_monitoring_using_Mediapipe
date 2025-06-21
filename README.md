

# Driver Monitoring System

This project uses MediaPipe and OpenCV to monitor a driver's state, specifically checking:
- Eye status (open/closed)
- Hand presence on the steering wheel
- Gaze direction (looking straight, left, right, up, or down)

## Prerequisites

Before running the application, ensure you have Python 3 installed. Then, install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## How to Run

1.  **Save the code:** Save the provided Python code as `driver_monitoring.py`.
2.  **Install dependencies:** Make sure you have `requirements.txt` in the same directory and install the dependencies as mentioned in the Prerequisites section.
3.  **Run the script:** Open a terminal or command prompt, navigate to the directory where you saved the file, and run the script:

    ```bash
    python driver_monitoring.py
    ```

    This will open a window displaying the live camera feed with annotations.

## Input

The system uses your computer's default webcam as input. Ensure your webcam is properly connected and accessible.

## Output

The output is displayed in a live video feed window, showing:

*   **Annotated facial landmarks and hand landmarks:** Visual overlays on the driver's face and hands.
*   **Textual status indicators:**
    *   `EYES CLOSED` or `EYES OPEN`: Indicates the driver's eye state.
    *   `HAND ON WHEEL`, `HAND OFF WHEEL`, or `NO HAND DETECTED`: Indicates whether a hand is detected on the simulated steering wheel area.
    *   `Looking Straight`, `Looking Left`, `Looking Right`, `Looking Up`, or `Looking Down`: Indicates the driver's gaze direction based on head pose.

To exit the application, press the `Esc` key.





## Important Note on Webcam Access

This application requires access to a webcam. In some sandboxed or virtual environments, direct webcam access might be restricted, leading to errors like `can't open camera by index`. To run this application successfully, ensure you are in an environment with proper webcam access.



    *   `YAWNING!` (if detected): Indicates if the driver is yawning or has their mouth open for an extended period.


