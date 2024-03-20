import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nsfw_detector import predict
import os
import psycopg2
import random
import schedule
import time
import json
import pdb
from transformers import pipeline
from PIL import Image

class Main:
    directory_path = "samples"
    screen_analysis_table = "image_analysis"
    screen_report_table = "image_report"
    file_extension = ".png"
    nsfw_threshold = 0.0048
    film_threshold = 0.75
    prev_threshold = 0.02
    prev_threshold2 = 0.01
    dictionary = {
        "hentai": "성관련",
        "sexy": "성관련",
        "porn": "성관련",
        "film": "영화",
        "betting": "도박",
    }

    def __init__(self):
        self.model = predict.load_model("./nsfw_mobilenet2.224x224.h5")
        self.betting_model = load_model('model_apps')
        self.film_pipe = pipeline("image-classification", model="pszemraj/beit-large-patch16-512-film-shot-classifier")
        # self.film_pipe = pipeline("image-classification", model="pszemraj/dinov2-small-film-shot-classifier")

    def test_model(self):
        file_list = os.listdir(self.directory_path)
        for filename in file_list:
            if not filename.endswith(self.file_extension):
                continue
            full_path_name = f'{self.directory_path}/{filename}'
            image = Image.open(full_path_name)
            film_result = self.film_pipe(image)[0]
            print(filename, '~~~~', film_result)

    def demo_process(self):
        print("Start demo...")

        file_list = os.listdir(self.directory_path)
        for idx in range(1, 5):
            # screen[1] file path
            filename = random.choice(file_list)
            if not filename.endswith(self.file_extension):
                continue
            
            full_path_name = f'{self.directory_path}/{filename}'
            image = Image.open(full_path_name)

            nsfw_result = predict.classify(self.model, full_path_name)
            film_result = self.film_pipe(image)[0]
            betting_result = self.check_betting(full_path_name)
            self.insert_data_into_db([idx, 1, 1, 'path'], nsfw_result, film_result, betting_result)

    def start_process(self):
        print("Start processing...")

        screen_list = self.fetch_screens()
        file_list = os.listdir(self.directory_path)
        for screen in screen_list[::-1]:
            # screen[1] file path
            filename = random.choice(file_list)
            if not filename.endswith(self.file_extension):
                continue

            full_path_name = f'{self.directory_path}/{filename}'
            image = Image.open(full_path_name)

            nsfw_result = predict.classify(self.model, full_path_name)
            film_result = self.film_pipe(image)[0]
            betting_result = self.check_betting(full_path_name)
            
            self.insert_data_into_db(screen, nsfw_result, film_result, betting_result)

    def check_betting(self, full_path_name):
        class_names = {0: "betting", 1: "others"}
        model = load_model('model_apps')
        img = load_img(full_path_name, target_size=(180, 180))
        img_array = tf.expand_dims(img, axis=0)
        predictions = self.betting_model.predict(img_array)
        pred_label = tf.argmax(predictions, axis = 1)
        print(class_names[pred_label.numpy()[0]])
        return 1.0 - pred_label.numpy()[0]
    
    def fetch_prev_records(self, client_id, laptop_id):
        result = [[0], [0]]
        try:
            print(f"Loading 2 prev records from {self.screen_analysis_table} table...")
            sql = f'''
                SELECT film FROM {self.screen_analysis_table} WHERE client_id = {client_id} and laptop_id = {laptop_id} ORDER BY created_at DESC LIMIT 2
            '''
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            print(f"Loaded {len(result)} records.")
        except Exception as e:
            print(f"Couldn't load the records: {e}")
        
        result += [[0], [0]]

        additional_value = result[0][0] * self.prev_threshold + result[1][0] * self.prev_threshold2
        print(client_id, laptop_id, "additional_value: ", additional_value)
        return additional_value
    
    def fetch_screens(self):
        latest_screen_id = 0
        try:
            print(f"Loading the latest screen ID from {self.screen_analysis_table}...")
            sql = f'''
                SELECT screen_id FROM {self.screen_analysis_table} ORDER BY screen_id DESC LIMIT 1
            '''
            self.cursor.execute(sql)
            latest_screen = self.cursor.fetchone()
            latest_screen_id = latest_screen[0]
        except Exception as e:
            print(f"Couldn't load the latest screen: {e}")

        result = []
        try:
            print("Loading new screens from screens table...")
            sql = f'''
                SELECT id, client_id, laptop_id, location FROM screens WHERE id > {latest_screen_id} ORDER BY created_at DESC LIMIT 10
            '''
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            print(f"Loaded {len(result)} new screens.")
        except Exception as e:
            print(f"Couldn't load the new screens: {e}")

        return result

    def insert_data_into_db(self, screen, nsfw_result, film_result={}, betting_result={}):
        try:
            additional_value = self.fetch_prev_records(screen[1], screen[2])
            output_records = []
            report_records = []
            film_score = film_result['score'] + additional_value
            if film_result['label'] not in ['extremeLongShot', 'fullShot', 'longShot', 'mediumCloseUp', 'mediumShot']:
                film_score = 0

            for key, value in nsfw_result.items():
                report_value = {}
                output_status = "safe"

                for v_key, v_value in value.items():
                    if v_value > self.nsfw_threshold and v_key in ['hentai', 'sexy', 'porn']:
                        report_value[v_key] = v_value

                if film_score > self.film_threshold:
                    report_value['film'] = film_score

                if betting_result == 1:
                    report_value['betting'] = betting_result

                if report_value != {}:
                    result_ko = []
                    result_en = []
                    output_status = "unsafe"
                    for r_key, r_val in report_value.items():
                        result_ko.append(f"{self.dictionary[r_key]}: {round(r_val, 4)}")
                        result_en.append(f"{r_key}: {round(r_val, 4)}")

                    report_records.append((
                        screen[0], 
                        screen[1], 
                        screen[2], 
                        json.dumps(report_value),
                        ', '.join(result_ko),
                        ', '.join(result_en)
                    ))

                output_records.append((
                    screen[0], 
                    screen[1], 
                    screen[2], 
                    value['drawings'], 
                    value['neutral'], 
                    value['hentai'], 
                    value['sexy'],
                    value['porn'],
                    film_score,
                    betting_result,
                    output_status
                ))
            
            output_sql = f'''
                INSERT INTO {self.screen_analysis_table} (
                    screen_id,
                    client_id,
                    laptop_id, 
                    drawings,
                    neutral,
                    hentai,
                    sexy,
                    porn,
                    film, 
                    betting, 
                    status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            self.cursor.executemany(output_sql, output_records)

            if len(report_records) > 0:
                report_sql = f'''
                    INSERT INTO {self.screen_report_table} (
                        screen_id,
                        client_id,
                        laptop_id,
                        result,
                        ko,
                        en
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                '''
                self.cursor.executemany(report_sql, report_records)

            self.conn.commit()
        except Exception as e:
            print(f"Couldn't insert the record into the database: {e}")

    def create_db_connection(self):
        try:
            self.conn = psycopg2.connect(
                host="localhost",
                database="postgres",
                user="postgres",
                password="rootroot",
                port="5432"
            )
            print("Connecting the database...")
            # self.conn = psycopg2.connect(
            #     host = "192.168.2.24",
            #     database = "production",
            #     user="postgres",
            #     password = "postgrespassword",
            #     port = "5432",
            # )
            self.cursor = self.conn.cursor()

            table_creation = f'''
                CREATE TABLE IF NOT EXISTS {self.screen_analysis_table} (
                    id SERIAL PRIMARY KEY,
                    screen_id INT,
                    client_id INT,
                    laptop_id INT,
                    drawings FLOAT,
                    neutral FLOAT,
                    hentai FLOAT,
                    sexy FLOAT,
                    porn FLOAT,
                    film FLOAT,
                    betting FLOAT,
                    status VARCHAR(30),
                    created_at TIMESTAMP default current_timestamp
                )
            '''
            self.cursor.execute(table_creation)

            report_table_creation = f'''
                CREATE TABLE IF NOT EXISTS {self.screen_report_table} (
                    id SERIAL PRIMARY KEY,
                    screen_id INT,
                    client_id INT,
                    laptop_id INT,
                    result JSON,
                    ko VARCHAR(255),
                    en VARCHAR(255),
                    created_at TIMESTAMP default current_timestamp,
                    approved_by INT,
                    approved_at TIMESTAMP,
                    approved_status VARCHAR(30)
                )
            '''
            self.cursor.execute(report_table_creation)
            self.conn.commit()

        except Exception as e:
            print(f"Couldn't connect the database: {e}")
            exit(1)

    def close_db_connection(self):
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    main = Main()
    main.create_db_connection()
    # main.start_process()
    main.demo_process()
    # main.close_db_connection()
    # main.test_model()

# if __name__ == "__main__":
#     main = Main()
#     main.create_db_connection()
#     schedule.every(60).seconds.do(main.start_process)
#     # schedule.every(1).minutes.do(main.start_process)
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
