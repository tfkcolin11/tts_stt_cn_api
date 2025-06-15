# Chinese Speech-to-Text (STT) and Text-to-Speech (TTS) API

## Description
This project provides a robust and easy-to-use API for converting Chinese text to speech (TTS) and speech to text (STT). It is built using FastAPI and leverages state-of-the-art open-source models for high-quality audio and transcription. The application is containerized with Docker for easy deployment and portability.

## Features
*   **Text-to-Speech (TTS):** Convert Chinese text into speech.
*   **Speech-to-Text (STT):** Transcribe Chinese audio files into text.
*   **FastAPI:** Modern, fast (high-performance) web framework for building APIs.
*   **Dockerized:** Easy to set up and run in any environment supporting Docker.

## Technologies Used
*   **Python 3.10:** The core programming language.
*   **FastAPI:** High-performance web framework for building the API.
*   **Uvicorn:** An ASGI server for running the FastAPI application.
*   **Hugging Face Transformers:** Utilized for the Speech-to-Text (STT) functionality with the Whisper model.
*   **Coqui TTS:** Used for the Chinese Text-to-Speech (TTS) functionality.
*   **PyTorch, Torchaudio, Torchvision:** Core deep learning libraries for model inference (CPU versions are used by default).
*   **Docker:** For containerizing the application and its dependencies.
*   **ffmpeg:** A multimedia framework used for audio processing within the Docker container.

## Installation

### Prerequisites
Ensure you have Docker installed on your system. You can download it from [Docker's official website](https://www.docker.com/get-started/).

### Building the Docker Image
Navigate to the root directory of this project in your terminal and build the Docker image:

```bash
docker build -t tts-stt-cn-api:latest .
```

### Running the Docker Container
Once the image is built, you can run the container. This will map port `8000` of the container to port `8000` on your host machine.

```bash
docker run -p 8000:8000 tts-stt-cn-api:latest
```
The API will be accessible at `http://localhost:8000`.

## API Endpoints

The API provides the following endpoints:

*   **`/` (GET):**
    *   **Description:** Root endpoint providing the API status and information about the loaded models.
    *   **Response:** JSON object with `message`, `stt_status`, `tts_status`, `stt_model_name`, `tts_model_name`, `pytorch_version`, `pytorch_location`, and `pytorch_get_default_device_attr`.

*   **`/api/tts` (POST):**
    *   **Description:** Converts Chinese text to speech.
    *   **Request Body:** JSON object with a `text` field.
        ```json
        {
          "text": "你好，世界！"
        }
        ```
    *   **Response:** An `audio/wav` file containing the generated speech.

*   **`/api/stt` (POST):**
    *   **Description:** Transcribes an audio file containing Chinese speech to text.
    *   **Request Body:** `multipart/form-data` with an `audio_file` field.
    *   **Response:** JSON object with the transcribed text.
        ```json
        {
          "text": " transcribed_text_here "
        }
        ```

*   **`/docs` (GET):**
    *   **Description:** Interactive OpenAPI (Swagger UI) documentation for all API endpoints.

## Usage Examples

### Text-to-Speech (TTS) Example
To convert text to speech, send a POST request to `/api/tts`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "肠炎, 寂寞，赶，阴面，阳面，念，于是，熟，油腻，效果，建筑，家庭，光，交响乐，下载，首，也许，古典，名曲，民歌，流行，歌曲，遥远，逛，夜，产生，画册，研究，改革，开放，一切，速度，避暑，经营，发财，捡，未婚夫，未婚妻，将来，送行，一路平安，研讨会，研讨，问好，捎，辛苦，麻烦，趟，展览，展览馆，大约，大使馆，大使"}' http://localhost:8000/api/tts --output speech.wav
```
This command will save the generated speech as `speech.wav` in your current directory.

### Speech-to-Text (STT) Example
To transcribe an audio file, send a POST request to `/api/stt`. Replace `your_audio_file.wav` with the path to your audio file.

```bash
curl -X POST -F "audio_file=@your_audio_file.wav" http://localhost:8000/api/stt
```
This command will return a JSON response containing the transcribed text.

## Contribution
Contributions are welcome! Please feel free to open issues or submit pull requests.

## License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
