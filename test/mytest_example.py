import os
import json
import time
import sys
 
# Get the parent directory
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
 
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from src.myNLI import FactChecker

PATH_SAVE = "temp/results"

if __name__ == "__main__":
    # input
    ls_claim = [
        "Đường được tạo ra từ cây mía",
        "Phần lớn trái chủ của lô trái phiếu 300 triệu USD đồng ý cho Novaland trả lãi chậm và đổi thành cổ phiếu NVL với giá ban đầu 40.000 đồng.",
        "Filip Nguyễn không đủ điều kiện dự Asian Cup 2019",
        "Sáng nay 13.12, Tổng Bí thư, Chủ tịch nước Trung Quốc Tập Cận Bình đã đến đặt vòng hoa và vào Lăng viếng Chủ tịch Hồ Chí Minh."
    ]
    
    t_0 = time.time()
    fact_checker = FactChecker()
    t_load = time.time() - t_0
    print("time load model: {}".format(t_load))
    claim = ls_claim[0]
    result = fact_checker.predict_nofilter(claim)
    print (result)
    # print("----- NO FILTER -----")
    # fact_checker.predict_nofilter(claim)
