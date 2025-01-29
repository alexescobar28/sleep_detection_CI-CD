import requests
import time
import cv2
import os

def test_api():
    # URL de la API
    base_url = 'http://localhost:5000'
    
    # Resetear el estado antes de empezar
    print("Reseteando el estado...")
    try:
        requests.post(f"{base_url}/reset")
    except requests.exceptions.RequestException as e:
        print(f"Error al resetear el estado: {e}")
        return
        
    # Pedir al usuario la ruta de la imagen
    image_path = input("Por favor, ingresa la ruta completa de tu imagen (por ejemplo: C:/Users/Alex/Desktop/imagen.jpg): ")
    
    if not os.path.exists(image_path):
        print(f"Error: No se puede encontrar el archivo en la ruta: {image_path}")
        return
    
    try:
        image_closed = cv2.imread(image_path)
        if image_closed is None:
            print(f"Error: No se pudo cargar la imagen")
            return
            
        _, img_encoded = cv2.imencode('.jpg', image_closed)
        
        print("Enviando frames...")
        for i in range(100):
            files = {
                'frame': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            
            try:
                response = requests.post(f"{base_url}/process_frame", files=files)
                response.raise_for_status()
                result = response.json()
                print(f"Frame {i}: {result}")
                
                # Mostrar información adicional de debug
                if result.get('is_microsleep'):
                    print(f"  ¡Microsueño detectado! Contador aux: {result.get('aux_counter')}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error al enviar frame {i}: {e}")
                break
                
            time.sleep(0.033)  # Simular 30 FPS

    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    test_api()