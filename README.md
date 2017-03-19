# attendance_check
Marks classroom attendance using OpenCV face detection and recognition

# Usage
1. Download and build [OpenCV 3.0+][1]

`git clone https://github.com/Eric4Jiang/attendance_check.git`

2. Make your dataset of students. Images will be saved in ".png" format

`python3 makeDataset.py -pata/to/save/images -path/to/haarcascades_frontface`

3. Check attendance

`python3 AttendanceCheck.py -path/to/dataset -path/to/haarcasecades_frontalface`

[1]: http://opencv.org/downloads.html
