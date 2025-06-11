# Insulator Defect Detection with Modified YOLOv8
Muchamad Arfim Muzaki. 2025. Development of The YOLOv8n Algorithm for Surface Insulator Defect Detection. Prof. Dr. Subiyanto, S.T., M.T. Program of Electrical Engineering Education, Faculty of Engineering, Universitas Negeri Semarang.

This repository contains a modified version of YOLOv8 using a custom module for insulator defect detection.

## Dataset
Google Drive download link：https://drive.google.com/file/d/1CPLCAwbD0cKcejCxESrrDicPeUmfrBcj/view?usp=drive_link

## Installation

```bash
git clone https://github.com/Arfim27/Insulator-Defect-Detection.git
cd Insulator-Defect-Detection
pip install -r requirements.txt

## Train
```bash
python detect.py

## Validation
yolo val model="your model" data= "your dataset"


