# YzupProje

Projeyi kendi bilgisyarınızda çalıştırmak için aşağıdaki komutları sırayla çalıştırınız:

`git clone https://github.com/hamza37yavuz/YzupProje.git`

Terminalden app.py dosyasının bulunduğu dizine gidiniz. Sonrasında aşağıdaki komutlardan birini çalıştırmanız yeterli olacaktır:

`python -m streamlit run .\app.py`

`streamlit run .\app.py`

Eğer Docker kullanarak projeyi çalıştırmak isterseniz app.py dosyasının bulunduğu dizine gidiniz sonrasında aşağıdaki komutları çalıştırınız:

`docker build -t my_streamlit_app . `

`docker run -p 8501:8501 my_streamlit_app`

Komutları çalıştırdıktan sonra:

[localhost](http://localhost:8501/) linkinden projeye erişebileceksiniz.
