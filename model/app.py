from flask import Flask, jsonify, request, send_file
import base64
from PIL import Image
import io
import api
import config as cfg
import pandas as pd
import os



class MLModelAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.route('/predict', methods=['POST'])(self.predict)
        
        # Serve training images by relative path (expects dataset locally)
        self.app.route('/images/<path:path>', methods=['GET'])(self.get_image)


        
    def get_image(self, path):
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

        # Extract the file extension
        file_extension = path.rsplit('.', 1)[1].lower() if '.' in path else None

        # Check if the file extension is allowed
        if file_extension and file_extension in allowed_extensions:
            full_path = os.path.abspath(
                os.path.join(str(cfg.DATASET_ROOT), path)
            )
            if os.path.exists(full_path):
                return send_file(full_path, mimetype=f'image/{file_extension}')
            else:
                return "File does not exist", 404
        else:
            return "Invalid image file", 400


    
    


    def predict(self):
        try:
            request_data = request.get_json()
            image_data = request_data.get('image', '')
            if not isinstance(image_data, str): 
                raise ValueError('Image data should be a base64-encoded string.')

            decoded_image = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(decoded_image))
            pil_image = pil_image.resize((224,224))
    
            prediction = api.search(cfg.MODELS, pil_image, cfg.WILL_RETURN_IMAGE_COUNT)

            meta = pd.read_csv(cfg.IMAGES_PATH_DF)

            if type(prediction) is str:
                return meta
            
            else:
                img_label = prediction["foundedImage"][0]
                print(img_label)
                count_similar_image = prediction["foundedImage"][0]
                print(count_similar_image)


                name = meta[meta["LABELS"] == img_label]["NAMES"].values[0]
                paths = meta[meta["LABELS"] == img_label]["PATHS"].values[:15]
                
                    
                print(name)
                print(paths)
                list_path = []
                for path in paths:
                    print("Datam", path)
                    list_path.append(path),
                
                
                print("list_path", list_path)
                    
                    
                return jsonify({'img_label': str(img_label),
                                "founded_similar_image_count":str(count_similar_image),
                                "species_name":name, 
                                "paths": list_path

                                }) 

                


            

        except ValueError as ve:
            return jsonify({'error': str(ve)})
      

    def run(self, debug=False):
        self.app.run(debug=debug)

if __name__ == '__main__':
    
    model_api = MLModelAPI()
    model_api.run(debug=True)
