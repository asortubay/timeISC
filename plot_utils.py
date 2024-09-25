import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

def plot_epoch_losses(results):
    fig, ax = plt.subplots(figsize=(10, 6))  # Step 1: Create a figure and axes object
    plt.plot(results['epoch_train_losses'], label='Training Loss', color='blue')
    plt.plot(results['epoch_val_losses'], label='Validation Loss', color='orange')
    # Find the epoch with the best validation loss
    best_val_loss_epoch = results['epoch_val_losses'].index(min(results['epoch_val_losses']))
    # Plot the best validation loss
    ax.scatter(best_val_loss_epoch, results['best_val_loss'], color='green', s=100, label='Best Validation Loss', zorder=5)
    ax.text(best_val_loss_epoch, results['best_val_loss'], f'{results["best_val_loss"]:.4f}', color='green', verticalalignment='bottom', horizontalalignment='right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss per Epoch')
    ax.legend()
    
    return fig

# display the results
def display_results(results, plot_type='val'):
    print(f"Train R^2: {results['R2_train']:.4f}")
    print(f"Validation R^2: {results['R2_val']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))  # Step 1: Create a figure and axes object
    if plot_type == 'train':
        y_true = np.ravel(results['y_train'])
        y_pred = np.ravel(results['yest_train'])
        title = 'Train'
        r2_score = results['R2_train']
        mse_score = results['MSE_train']
    elif plot_type == 'test':
        print(f"Test R^2: {results['R2_test']:.4f}")
        print(f"Test (train subjects) R^2: {results['R2_test_trainsubs']:.4f}")
        y_true = np.ravel(results['y_test'])
        y_pred = np.ravel(results['yest_test'])
        mse_score = results['MSE_test']
        title = 'Test'
        r2_score = results['R2_test']
    elif plot_type == 'test_trainsubs':
        print(f"Test R^2: {results['R2_test']:.4f}")
        print(f"Test (train subjects) R^2: {results['R2_test_trainsubs']:.4f}")
        y_true = np.ravel(results['y_test_trainsubs'])
        y_pred = np.ravel(results['yest_test_trainsubs'])
        r2_score = results['R2_test_trainsubs']
        mse_score = results['MSE_test_trainsubs']
        title = 'Test on Train Subs'
    else:  # Default to 'val'
        y_true = np.ravel(results['y_val'])
        y_pred = np.ravel(results['yest_val'])
        title = 'Validation'
        r2_score = results['R2_val']
        mse_score = results['MSE_val']

    ax.scatter(y_true, y_pred, color='blue', label=f'True vs. Predicted ISC ({title})')
    
    # Fit a line to the scatter plot
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), "r--", label='Fit Line')
    ax.plot(y_true, y_true, "g--", label='Perfect fit line')
    
    ax.set_xlabel('True ISC')
    ax.set_ylabel('Predicted ISC')
    ax.set_title(f'True vs. Predicted ISC Values. {title} R^2: {r2_score:.4f} MSE: {mse_score:.4f}')
    ax.legend()
    
    return fig
