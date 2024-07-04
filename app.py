import os
import whisper
import openai
from flask import Flask, request, jsonify, send_file
from fpdf import FPDF
from pydub import AudioSegment

app = Flask(__name__)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializar el modelo de Whisper
model = whisper.load_model("base")
openai.api_key = openai_api_key

@app.route('/')
def index():
    return "Servidor funcionando"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Guardar el archivo temporalmente
        temp_file_path = os.path.join(base_dir, 'temp_audio.wav')
        file.save(temp_file_path)
        print("Archivo guardado temporalmente en:", temp_file_path)

        # Convertir el audio a mono y obtener la tasa de muestreo
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_channels(1)  # Convertir a mono
        audio = audio.set_frame_rate(16000)  # Ajustar la tasa de muestreo si es necesario

        # Guardar el audio convertido
        mono_file_path = os.path.join(base_dir, 'temp_audio_mono.wav')
        audio.export(mono_file_path, format="wav")
        print("Archivo de audio convertido guardado en:", mono_file_path)

        # Transcribir el audio usando Whisper
        result = model.transcribe(mono_file_path)
        transcript = result["text"]
        print("Transcripción obtenida:", transcript)

        # Formatear la transcripción como una historia clínica usando GPT-3.5-turbo
        historia_clinica = formatear_historia_clinica(transcript)
        print("Historia clínica generada:", historia_clinica)

        pdf_path = generar_pdf(historia_clinica)
        print("PDF generado en:", pdf_path)

        return jsonify({'transcript': transcript, 'pdf_path': pdf_path})

    except Exception as e:
        print("Error durante la transcripción:", str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        # Eliminar los archivos temporales
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(mono_file_path):
            os.remove(mono_file_path)

def formatear_historia_clinica(transcript):
    prompt = f"""
    Eres un asistente médico encargado de convertir conversaciones en historias clínicas bien estructuradas. A continuación se presenta una conversación entre un médico y un paciente. Por favor, convierte esta conversación en una historia clínica detallada siguiendo el formato proporcionado.

    Conversación:
    "{transcript}"

    Historia Clínica:
    Paciente: [Nombre del paciente]
    Edad: [Edad del paciente]
    Fecha: [Fecha de la consulta]

    Motivo de consulta:
    [Describir el motivo de la consulta basado en la conversación]

    Interrogatorio:
    [Preguntas y respuestas relevantes extraídas de la conversación]

    Observaciones:
    [Observaciones adicionales sobre el estado del paciente]

    Diagnóstico preliminar:
    [Diagnóstico basado en la información proporcionada]

    Plan de tratamiento:
    [Descripción del tratamiento recomendado]

    Recomendaciones:
    [Cualquier recomendación adicional para el paciente]

    Seguimiento:
    [Plan de seguimiento si es necesario]
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente que convierte conversaciones a historias clínicas."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )

    historia_clinica = response['choices'][0]['message']['content'].strip()
    return historia_clinica

def generar_pdf(historia_clinica):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Historia Clínica', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(10)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("Historia Clínica")
    pdf.chapter_body(historia_clinica)

    pdf_path = os.path.join(base_dir, 'historia_clinica.pdf')
    pdf.output(pdf_path)
    return pdf_path

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_path = request.args.get('pdf_path')
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    else:
        return jsonify({'error': 'PDF no encontrado'}), 404

if __name__ == '__main__':
    # print(f"El archivo de credenciales está en: {credentials_path}")
    app.run(debug=True, host='0.0.0.0', port=5000)
