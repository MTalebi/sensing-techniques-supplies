# Train GAN on limited damage data
real_damage_signals = load_damage_data()  # Only 50 samples!

# Generate synthetic damage data
gan = WassersteinGAN(input_dim=1024)
gan.train(real_damage_signals, epochs=5000)

# Generate 1000 synthetic damage samples
synthetic_damage = gan.generate(n_samples=1000)

# Combine for balanced training set
healthy_data = load_healthy_data()  # 10000 samples
augmented_damage = np.vstack([real_damage_signals, 
                             synthetic_damage])

# Train classifier on balanced dataset
classifier = SVM()
X = np.vstack([healthy_data, augmented_damage])
y = np.hstack([np.zeros(len(healthy_data)), 
               np.ones(len(augmented_damage))])
classifier.fit(X, y) 