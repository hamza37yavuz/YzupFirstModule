# National Technology Academy Mini Project

This project was developed as part of the **National Technology Academy Artificial Intelligence Specialization Training**.

## How to Run

To run the project on your own computer, execute the commands below in order:

1. Open a terminal or command prompt.

2. Clone the project using the following command:
   `git clone https://github.com/hamza37yavuz/YzupProje.git`

3. Navigate to the directory where `app.py` is located:
   `cd YzupProje`

4. Install the required Python libraries:
   `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

5. Then run the following command to open the Streamlit app:
   `python -m streamlit run .\app.py` or `streamlit run .\app.py`

## Running with Docker

If you want to run the project using Docker, go to the directory containing the `Dockerfile` and run these commands in order:

* `docker build -t my_streamlit_app .`
* `docker run -p 8501:8501 my_streamlit_app`

After running the commands, you can access the project at:

* `http://localhost:8501/`
