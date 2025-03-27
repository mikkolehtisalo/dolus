import tensorflow as tf
import tools
import argparse

def main():
    parser = argparse.ArgumentParser(prog='run.py', description='Runs the pre-trained neural network with Tensorflow')
    parser.add_argument('-l', '--length', help='padding to length', default=64) 
    parser.add_argument('model', help="Trained model file")
    parser.add_argument('input')
    parser.parse_args()
    args = parser.parse_args()

    input = args.input
    model = tf.keras.models.load_model(args.model)
    normalized =  tools.preprocess_password(input, args.length)
    normalized = normalized.reshape((1, args.length, 1))
    prediction = model.predict(normalized)

    threshold = 0.5  
    if prediction >= threshold:
        print(f"The string '{input}' is classified as GOOD.")
    else:
        print(f"The string '{input}' is classified as BAD.")

if __name__ == "__main__":
    main()
