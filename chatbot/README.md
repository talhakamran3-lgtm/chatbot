# Chatbot Project

This project implements a chatbot using natural language processing and machine learning techniques. The chatbot is trained to understand user inputs and respond appropriately based on predefined intents.

## Project Structure

```
chatbot
├── src
│   ├── train.py          # Contains the training logic for the chatbot
│   ├── chatbot.py        # Main logic for processing user input and generating responses
│   └── ui
│       ├── app.py       # Entry point for the user interface
│       └── templates
│           └── index.html # HTML template for the chatbot's web interface
├── intents.json          # Defines patterns and responses for various user inputs
├── words.pkl             # Serialized list of unique words used in training data
├── classes.pkl           # Serialized list of unique classes (tags) corresponding to intents
├── chatbot_model.h5      # Trained Keras model for the chatbot
├── requirements.txt      # Lists Python dependencies required for the project
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd chatbot
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Train the model**:
   Run the training script to create the model:
   ```
   python src/train.py
   ```

4. **Run the chatbot**:
   Start the user interface by running:
   ```
   python src/ui/app.py
   ```

5. **Access the chatbot**:
   Open your web browser and navigate to `http://localhost:5000` to interact with the chatbot.

## Usage Guidelines

- Type your queries in the input field and press enter to receive responses from the chatbot.
- The chatbot is designed to handle various intents defined in the `intents.json` file.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.