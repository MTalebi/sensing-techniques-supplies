from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_detector(detector, test_data, test_labels):
    """Comprehensive evaluation of hybrid detector"""
    
    predictions = []
    recon_errors = []
    
    for sample in test_data:
        result = detector.detect_anomaly(sample)
        predictions.append(int(result['anomaly']))
        recon_errors.append(result['recon_error'])
    
    # Classification metrics
    print("Classification Report:")
    print(classification_report(test_labels, predictions))
    
    # ROC AUC using reconstruction error as score
    auc_score = roc_auc_score(test_labels, recon_errors)
    print(f"ROC AUC Score: {auc_score:.3f}")
    
    # Detection delay analysis
    damage_indices = np.where(test_labels == 1)[0]
    if len(damage_indices) > 0:
        first_damage = damage_indices[0]
        first_detection = np.where(
            np.array(predictions[first_damage:]) == 1
        )[0]
        
        if len(first_detection) > 0:
            delay = first_detection[0] * 5  # 5-min windows
            print(f"Detection delay: {delay} minutes")
    
    return predictions, recon_errors

# Evaluation on bridge test data
predictions, scores = evaluate_detector(
    detector, test_data, test_labels
)

# Visualize results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(test_labels)), test_labels, 'b-', 
         label='True Labels', linewidth=2)
plt.plot(range(len(predictions)), predictions, 'r--', 
         label='Predictions', linewidth=2)
plt.ylabel('Anomaly Label')
plt.legend()
plt.title('Hybrid ML Detector Performance')

plt.subplot(2, 1, 2)
plt.plot(range(len(scores)), scores, 'g-', linewidth=1)
plt.axhline(y=detector.threshold, color='r', 
           linestyle='--', label='Threshold')
plt.ylabel('Reconstruction Error')
plt.xlabel('Time Window')
plt.legend()
plt.tight_layout()
plt.show() 