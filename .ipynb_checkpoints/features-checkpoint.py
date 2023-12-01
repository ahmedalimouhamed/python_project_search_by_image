{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4d028506-aa01-4a5b-b373-9a1000209c73",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtensorflow\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mtf\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtensorflow\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapplications\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ResNet50\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtensorflow\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapplications\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mresnet50\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m preprocess_input\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.applications import ResNet50\n",
    "from tensorflow.keras.applications.resnet50 import preprocess_input\n",
    "from tensorflow.keras.layers import GlobalMaxPooling2D\n",
    "\n",
    "model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))\n",
    "model.trainable=False\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52a4066f-30e3-407b-b061-bf00eb0aa736",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2bb7920-b5d4-4152-b642-530a2f62e266",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
