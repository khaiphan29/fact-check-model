# automated-factchecking-vi
The responose time is around 15s to 60s

## Download additional files
Download the folder **mDeBERTa (ft) V6** at [link](https://drive.google.com/drive/folders/1B38WK4zcuI-i78pWXsuURbUYi0EDyDYe?usp=sharing). Then put the folder in src directory.

## How to setup environment
My Python Version is 9.8.x
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
## How to run test example (optional)

```
//test on the original pipeline
python test/test_example.py

//test on the updated pipeline
python test/mytest_example.py
```

# Activate the backend
Skip the first cmd if the venv has already been activated
```
source venv/bin/activate
uvicorn main:app --reload 
```

# How the API work
The **POST** request should be sent to URL http://127.0.0.1:8000/claim with the request body: 
```json
{
  "claim": "[YOUR_CLAIM]"
}
```

The response body **example**: \
STATUS CODE 200
```json
{
  "claim": "Tổng hợp tiền chất carbon và nitrogen nhằm tạo ra carbon nitride, hợp chất cứng hơn cubic boron nitride, hiện nay là vật liệu cứng thứ hai trên thế giới chỉ sau kim cương.",
  "final_label": 1,
  "evidence": "Một nhóm nhà khoa học quốc tế đứng đầu là các nhà nghiên cứu ở Trung tâm khoa học điều kiện cực hạn tại Đại học Edinburgh tạo ra đột phá mới khi tổng hợp tiền chất carbon và nitrogen nhằm tạo ra carbon nitride, hợp chất cứng hơn cubic boron nitride, hiện nay là vật liệu cứng thứ hai trên thế giới chỉ sau kim cương.\nKết quả phân tích hé lộ ba hợp chất carbon nitride tổng hợp có cấu trúc cần thiết đối với vật liệu siêu cứng.\nNgoài độ cứng, những hợp chất carbon nitride gần như không thể phá hủy này cũng có khả năng phát quang, áp điện và mật độ năng lượng cao, có thể lưu trữ lượng lớn năng lượng trong khối lượng nhỏ.\nDù giới khoa học nhận thấy tiềm năng của carbon nitride từ thập niên 1980, bao gồm khả năng chịu nhiệt cao, việc tạo ra chúng là một câu chuyện khác.\nNhóm nghiên cứu bao gồm nhiều chuyên gia vật liệu từ Đại học Bayreuth, Đức, và Đại học Linköping, Thụy Điển, đạt được thành tựu khi để các dạng khác nhau của tiền chất carbon nitrogen chịu áp suất 70 - 135 gigapascal (gấp khoảng một triệu lần áp suất khí quyển), đồng thời nung nóng chúng tới hơn 1.500 độ C. Sau đó, họ kiểm tra sắp xếp nguyên tử thông qua chùm tia X ở Cơ sở nghiên cứu Synchrotron châu Âu tại Pháp, Deutsches Elektronen - Synchrotron tại Đức và Advanced Photon Source tại Mỹ.",
  "provider": "vnexpress.net",
  "url": "https://vnexpress.net/vat-lieu-co-the-soan-ngoi-kim-cuong-ve-do-cung-4688566.html"
}
```
STATUS CODE 204 - NO CONTENT
```json
{}
```
NOTE for the label_code:
+ 0 === "Tin giả"
+ 1 === "Tin xác thực"
+ 2 === "Không xác định"