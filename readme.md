
### Service
sudo mv image_analysis.service /etc/systemd/system/image_analysis.service
sudo systemctl daemon-reload
sudo systemctl start image_analysis.service
sudo systemctl enable image_analysis.service
sudo systemctl status image_analysis.service


## Install

Python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt


### Run scripts
python3 run_me.py
python3 train.py
