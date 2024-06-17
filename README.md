# HexML: Intelligent Entertaining AI Assistant

HexML is an intelligent entertaining AI assistant designed to interact with users through both voice and text commands. It features mechanical eyes mounted on a laptop that can track the environment and engage in conversations with people. HexML is equipped with a dynamic 3D model displayed on a website with an admin panel for monitoring and control. This project integrates various technologies such as Arduino, Python, and Docker to create a versatile and modular assistant.

## Features

- **Interactive Communication**: HexML can interact with users in real-time using voice and text commands.
- **Voice Recognition and Synthesis**: Utilizes advanced speech-to-text and text-to-speech technologies for natural language interaction.
- **Natural Language Processing (NLP)**: Analyzes and understands user commands and generates meaningful responses.
- **Environmental Awareness**: Equipped with mechanical eyes and cameras to track movements and recognize objects and faces using computer vision.
- **Device Integration**: Connects and communicates with various devices and sensors via USB and other interfaces.
- **User Interface and Admin Panel**: A web-based interface for managing and configuring the assistant, including a 3D model visualization.
- **Modularity and Extensibility**: Designed to allow easy addition of new features and modules.
- **Autonomy and Energy Efficiency**: Optimized for efficient resource usage and autonomous operation.

## Technical Stack

- **Programming Languages**: Python, JavaScript
- **Frameworks and Libraries**:
  - NLP: spaCy, NLTK
  - Computer Vision: OpenCV, TensorFlow
  - Web Development: Flask or Django for backend, React or Vue.js for frontend
- **Hardware**: Arduino, Raspberry Pi, servos, cameras, microphone
- **Platforms**: Docker for containerization, cloud services for speech recognition and other functionalities

## Getting Started

### Prerequisites

- **Hardware**: Arduino, servos, cameras, Raspberry Pi (optional), 3D printer for mechanical parts
- **Software**: Docker, Python 3.9 or later, Node.js

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hexml.git
   cd hexml
   ```

2. **Build and run the Docker container**:
   ```bash
   docker-compose up --build
   ```

3. **Set up the Arduino**:
   - Connect the Arduino to your computer.
   - Upload the provided Arduino sketch from the `arduino` directory.

4. **Run the Python script**:
   ```bash
   python app/your_script.py
   ```

### Usage

- Access the web interface at `http://localhost:3000`.
- Interact with HexML via the web interface or through connected devices.

## Contributing

We welcome contributions from the community! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- Special thanks to the open-source community for providing the tools and libraries that made this project possible.
