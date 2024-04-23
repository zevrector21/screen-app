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
import shutil


class Main:
    directory_path = "files"
    screen_analysis_table = "image_analysis"
    screen_report_table = "image_report"
    file_extension = ".jpg"
    nsfw_threshold = {
        'hentai': 0.8,
        'sexy': 0.75,
        'porn': 0.75
    }
    film_threshold = 0.95
    prev_threshold = 0.005
    prev_threshold2 = 0.001
    dictionary = {
        "hentai": "성관련(1)",
        "sexy": "성관련(2)",
        "porn": "성관련(3)",
        "film": "영화",
        "betting": "도박",
    }
    betting_exception_list= ["mp4", "avi", "mpeg", "pdf"]

    def __init__(self):
        self.model = predict.load_model("./nsfw_mobilenet2.224x224.h5")
        # self.betting_model = load_model('model_apps')
        # self.film_pipe = pipeline("image-classification", model="pszemraj/beit-large-patch16-512-film-shot-classifier")
        # self.film_pipe = pipeline("image-classification", model="pszemraj/dinov2-small-film-shot-classifier") - primary

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
        for idx in range(1, 10):
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
        self.betting_model = load_model('model_apps')

        screen_list = self.fetch_screens()
        file_list = os.listdir(self.directory_path)
        for screen in screen_list[::-1]:
            try:
                # screen[1] file path
                # filename = random.choice(file_list)
                # if not filename.endswith(self.file_extension):
                #     continue

                # full_path_name = f'{self.directory_path}/{filename}'
                full_path_name = screen[4]
                image = Image.open(full_path_name)

                nsfw_result = predict.classify(self.model, full_path_name)
                betting_result = {}

                if screen[5].split('.')[-1].lower() not in self.betting_exception_list:
                    betting_result = self.check_betting(full_path_name)
                
                # film_result = self.film_pipe(image)[0]
                self.insert_data_into_db(screen, nsfw_result, {}, betting_result)
            except:
                pass

    def check_betting(self, full_path_name):
        class_names = {0: "betting", 1: "others", 2: "photo", 3: "windows"}
        img = load_img(full_path_name, target_size=(800, 600))
        img_temp = load_img(full_path_name)
        print(f"image resolution: {img_temp.width} * {img_temp.height}")
        if img_temp.height < 500 or img_temp.width < 600:
            return 3
        img_array = tf.expand_dims(img, axis=0)
        predictions = self.betting_model.predict(img_array)
        pred_label = tf.argmax(predictions, axis = 1)
        print(class_names[pred_label.numpy()[0]])
        return float(pred_label.numpy()[0])

    
    def fetch_prev_records(self, client_id, laptop_id):
        result = [[0], [0]]
        try:
            print(f"Loading 2 prev records from {self.screen_analysis_table} table...")
            sql = f'''
                SELECT film FROM {self.screen_analysis_table} WHERE client_id = {client_id} and laptop_id = {laptop_id} ORDER BY created_at DESC LIMIT 2
            '''
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            # print(f"Loaded {len(result)} records.")
        except Exception as e:
            # print(f"Couldn't load the records: {e}")
            pass
        
        result += [[0], [0]]

        additional_value = result[0][0] * self.prev_threshold + result[1][0] * self.prev_threshold2
        # print(client_id, laptop_id, "additional_value: ", additional_value)
        return additional_value
    
    def fetch_screens(self):
        result = []
        try:
            print("Loading new screens from screens table...")
            # sql = f"SELECT id, client_id, laptop_id, location, internal_path FROM screens WHERE id > {latest_screen_id} and type != 'main' ORDER BY created_at DESC"
            sql = f'SELECT s.id, s.client_id, s.laptop_id, s.location, s.internal_path, s.app_title FROM screens s LEFT JOIN screen_preprocesses sp ON s.id = sp.screen_id WHERE sp.is_image_processed = false and sp.ignore_image_processing = false ORDER BY s.created_at DESC'
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            print(f"Loaded {len(result)} new screens.")
        except Exception as e:
            print(f"Couldn't load the new screens: {e}")

        return result

    def insert_data_into_db(self, screen, nsfw_result, film_result={}, betting_result={}):
        try:
            # additional_value = self.fetch_prev_records(screen[1], screen[2])
            # additional_value = 0
            # film_score = film_result['score'] + additional_value
            # if film_result['label'] not in ['extremeLongShot', 'fullShot', 'longShot', 'mediumCloseUp', 'mediumShot']:
            film_score = 0
            output_records = []
            report_records = []

            for key, value in nsfw_result.items():
                report_value = {}
                output_status = "safe"

                for v_key, v_value in value.items():
                    if v_key not in ['hentai', 'sexy', 'porn']:
                        continue 
                    if v_value > self.nsfw_threshold[v_key]:
                        report_value[v_key] = v_value

                # if film_score > self.film_threshold:
                #     report_value['film'] = film_score

                if betting_result == 0:
                    shutil.copyfile(screen[4], f'files/temp/{screen[0]}.png')
                    report_value['betting'] = betting_result

                if report_value != {}:
                    result_ko = []
                    result_en = []
                    output_status = "unsafe"
                    for r_key, r_val in report_value.items():
                        result_ko.append(f"{self.dictionary[r_key]}")
                        result_en.append(f"{r_key}")
                        # result_ko.append(f"{self.dictionary[r_key]}: {round(r_val * 100, 2)}%")
                        # result_en.append(f"{r_key}: {round(r_val * 100, 2)}%")

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
            # self.conn = psycopg2.connect(
            #     host="localhost",
            #     database="postgres",
            #     user="postgres",
            #     password="rootroot",
            #     port="5432"
            # )
            print("Connecting the database...")
            self.conn = psycopg2.connect(
                host = "192.168.2.24",
                database = "production",
                user="postgres",
                password = "postgrespassword",
                port = "5432",
            )
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
    while True:
        main.start_process()
        time.sleep(300)
    # main.demo_process()
    main.close_db_connection()
    # main.test_model()

# if __name__ == "__main__":
#     main = Main()
#     main.create_db_connection()
#     schedule.every(60).seconds.do(main.start_process)
#     # schedule.every(1).minutes.do(main.start_process)
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
