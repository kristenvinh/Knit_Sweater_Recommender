# Add a temporary classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # Classify by category (Sweater, Hat, etc.)
])

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
# ... model.fit() on your knitwear data ...
model.save('custom_knitwear_resnet.h5')