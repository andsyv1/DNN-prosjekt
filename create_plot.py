import pandas as pd
import matplotlib.pyplot as plt

def plot_column_from_csv(file_path, column_name, column_name2, save_path, y_label, title):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Check if the specified column exists
        if column_name not in data.columns:
            print(f"Column '{column_name}' not found in the CSV file.")
            print("Available columns:", ", ".join(data.columns))
            return

        # Plot the specified column
        plt.figure(figsize=(10, 6))
        plt.plot(data[column_name], marker='o', linestyle='-', label="mAP50", color='b')
        plt.plot(data[column_name2], marker='o', linestyle='-', label="mAP50-95", color='r')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)

        
        # Save the plot as a .png file
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved as '{save_path}'.")

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "./dataset/runs/segment/train/results.csv"
    save_path = "./dataset/runs/segment/train/mAP50.png"
    column_name = "metrics/mAP50(B)"
    column_name2 = "metrics/mAP50-95(B)"
    y_label = "mAP"
    title = "mAP50 vs mAP50-95"
    plot_column_from_csv(file_path, column_name, column_name2, save_path, y_label, title)
