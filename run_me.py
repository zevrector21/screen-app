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


class Main:
    directory_path = "files"
    screen_analysis_table = "image_analysis"
    screen_report_table = "image_report"
    file_extension = ".png"
    threshold = 0.0048

    def __init__(self):
        self.model = predict.load_model("./nsfw_mobilenet2.224x224.h5")
        pass

    def test_model(self):
        class_names = {0: "skype", 1: "skack", 2: "vs_code"}
        model = load_model('model_apps')
        test_image = 'screens/skype/me.jpeg'   
        img = load_img(test_image, target_size=(180, 180))
        img_array = tf.expand_dims(img, axis=0)
        predictions = model.predict(img_array)
        pred_label = tf.argmax(predictions, axis = 1)
        print(predictions)
        print(class_names[pred_label.numpy()[0]])

    def demo_process(self):
        print("Start demo...")

        file_list = os.listdir(self.directory_path)
        for idx in range(1, 21):
            # screen[1] file path
            filename = random.choice(file_list)
            if not filename.endswith(self.file_extension):
                continue
            result = predict.classify(self.model, f'{self.directory_path}/{filename}')
            self.insert_data_into_db([idx, random.randint(1, 50), random.randint(1, 50), 'path'], result)

    def start_process(self):
        print("Start processing...")

        screen_list = self.fetch_screens()
        file_list = os.listdir(self.directory_path)
        for screen in screen_list:
            # screen[1] file path
            filename = random.choice(file_list)
            if not filename.endswith(self.file_extension):
                continue
            result = predict.classify(self.model, f'{self.directory_path}/{filename}')
            self.insert_data_into_db(screen, result)

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
                SELECT id, client_id, laptop_id, location FROM screens WHERE id > {latest_screen_id} ORDER BY created_at DESC
            '''
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            print(f"Loaded {len(result)} new screens.")
        except Exception as e:
            print(f"Couldn't load the new screens: {e}")
        
        return result

    def insert_data_into_db(self, screen, result):
        try:
            output_sql = f'''
                INSERT INTO {self.screen_analysis_table} (
                    screen_id,
                    drawings,
                    neutral,
                    hentai,
                    sexy,
                    porn,
                    status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''
            output_records = []
            report_records = []
            for key, value in result.items():
                report_value = {}
                output_status = "safe"

                for v_key, v_value in value.items():
                    if v_value > self.threshold and v_key in ['hentai', 'sexy', 'porn']:
                        report_value[v_key] = v_value

                if report_value != {}:
                    output_status = "unsafe"
                    report_records.append((
                        screen[0], 
                        screen[1], 
                        screen[2], 
                        json.dumps(report_value)
                    ))

                output_records.append((
                    screen[0], 
                    value['drawings'], 
                    value['neutral'], 
                    value['hentai'], 
                    value['sexy'],
                    value['porn'],
                    output_status
                ))

            self.cursor.executemany(output_sql, output_records)

            if len(report_records) > 0:
                report_sql = f'''
                    INSERT INTO {self.screen_report_table} (
                        screen_id,
                        client_id,
                        laptop_id,
                        result
                    ) VALUES (%s, %s, %s, %s)
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
                    drawings FLOAT,
                    neutral FLOAT,
                    hentai FLOAT,
                    sexy FLOAT,
                    porn FLOAT,
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
    main.start_process()
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
