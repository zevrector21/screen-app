import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
import pathlib 
import time
import schedule
import psycopg2
import shutil
import pdb


def main():

    screen_report_table = "image_report"

    try:
        print("Connecting the database...")
        conn = psycopg2.connect(
            host = "192.168.2.24",
            database = "production",
            user="postgres",
            password = "postgrespassword",
            port = "5432",
        )
        cursor = conn.cursor()

    except Exception as e:
        print(f"Couldn't connect the database: {e}")
        exit(1)

    try:
        sql = f'''
            SELECT screen_id FROM {screen_report_table} WHERE approved_status = 'REPORTED' and trained_status != true
        '''
        cursor.execute(sql)
        reported_result = cursor.fetchall()
    except Exception as e:
        print(f"Couldn't load reported screens: {e}")

    try:
        sql = f'''
            SELECT screen_id FROM {screen_report_table} WHERE approved_status = 'APPROVED' and trained_status != true
        '''
        cursor.execute(sql)
        approved_result = cursor.fetchall()
    except Exception as e:
        print(f"Couldn't load approved screens: {e}")

    print('Reported records:', len(reported_result))
    print('Approved records:', len(approved_result))

    if len(reported_result) > 0 or len(approved_result) > 0:
        for record in reported_result:
            try:
                shutil.move(f'files/temp/{record[0]}.png', f'screens/betting/{record[0]}.png')
            except:
                pass

        for record in approved_result:
            try:
                shutil.move(f'files/temp/{record[0]}.png', f'screens/others/{record[0]}.png')
            except:
                pass

    train_result = train()
    if train_result:
        try:
            sql = f'''
                UPDATE {screen_report_table} SET trained_status = true WHERE screen_id = %s
            '''
            updated_records = reported_result + approved_result
            cursor.executemany(sql, updated_records)
            conn.commit()

        except Exception as e:
            print(f"Couldn't update the screens: {e}")

        cursor.close()
        conn.close()

        print('Re-trained the model successfully!')
    else:
        print('Re-trained the model failed!, try again later.')

# =================================

def train():
    try:
        print("TensorFlow version:", tf.__version__)

        data_dir = "screens"

        # Training split 
        train_ds = tf.keras.utils.image_dataset_from_directory( 
            data_dir, 
            validation_split=0.2, 
            subset="training", 
            seed=123, 
            image_size=(800,600), 
            batch_size=32
        )

        # Testing or Validation split 
        val_ds = tf.keras.utils.image_dataset_from_directory( 
            data_dir, 
            validation_split=0.2, 
            subset="validation", 
            seed=123, 
            image_size=(800,600), 
            batch_size=32
        )
         
        class_names = train_ds.class_names 
        print(class_names)

         
        num_classes = len(class_names) 
          
        model = Sequential([ 
            layers.Rescaling(1./255, input_shape=(None, None, 3)), 
            layers.Conv2D(16, 3, padding='same', activation='relu'), 
            layers.MaxPooling2D(), 
            layers.Conv2D(32, 3, padding='same', activation='relu'), 
            layers.MaxPooling2D(), 
            layers.Conv2D(64, 3, padding='same', activation='relu'), 
            layers.MaxPooling2D(), 
            layers.GlobalMaxPooling2D(), 
            layers.Dense(128, activation='relu'), 
            layers.Dense(num_classes) 
        ])

        model.compile(optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy( 
                from_logits=True), 
            metrics=['accuracy']) 
        model.summary()

        epochs = 80
        history = model.fit( 
          train_ds, 
          validation_data=val_ds, 
          epochs=epochs 
        )

        model.save("model_apps_new")

        return True

    except Exception as e:
        print("Errors in train:", e)
        return False

if __name__ == "__main__":
    train()
    exit(0)
    local_timezone = "America/New_York"
    schedule.every().day.at("07:00", local_timezone).do(main)
    schedule.every().day.at("13:00", local_timezone).do(main)
    schedule.every().day.at("19:00", local_timezone).do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)
