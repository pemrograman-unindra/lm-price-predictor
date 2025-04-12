# Sistem Prediksi Harga Logam Mulia (LM) menggunakan metode neural network

Proyek ini bertujuan memprediksi harga logam mulia (emas batangan) berdasarkan data historis menggunakan model LSTM.

## Getting Started
1. Pastikan perangkat kamu sudah terinstall python3, python3-pip dan python3-venv
2. Buat Virtual Environment
```bash
python3 -m venv venv
```
3. Aktifkan Virtual Environment
```bash
# linux / mac
source venv/bin/activate

# windows
venv\Scripts\activate
```
4. Install dependensi
```bash
pip install numpy pandas scikit-learn matplotlib seaborn keras tensorflow
```
5. Perbarui file `data/lm_price.json` dari https://www.logammulia.com/id/grafik-harga-emas (lihat inspect element, cek response api https://www.logammulia.com/data-base-price/gold/sell?_token=xxx)
6. Konversi data json menjadi csv
```bash
python src/json_to_csv.py
```
7. Latih model
```bash
python src/train.py
```
8. Lakukan prediksi
```bash
python src/predict.py
```
