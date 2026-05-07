# Salient Object Detection (SOD) using Encoder-Decoder CNN

Quick Start: Usage Instructions

Ensure you have the required dependencies installed (PyTorch, OpenCV, Streamlit, etc.) before running the following scripts:
You can install them by using this command:
pip install torch torchvision torchaudio numpy Pillow matplotlib opencv-python scikit-learn tqdm  
pip install streamlit pillow torchvision    

To start the training process using the DUTS dataset:
python train.py

To evaluate the trained model on the test set and generate performance metrics:
python evaluate.py

To launch the web interface and test the model on your own images:
python -m streamlit run app.py
or: 
streamlit run app.py

Project Structure:
data_loader.py: Handles dataset augmentation, normalization, and PyTorch DataLoader integration for the DUTS dataset.
sod_model.py: Contains the custom Encoder-Decoder CNN architecture built from scratch.  
train.py: The main training script optimized for NVIDIA T4 GPU acceleration.
evaluate.py: Script used to calculate Mean IoU, Precision, and Recall on the test set.
app.py: The Streamlit interface for real-time model inference and visualization.

Project Documentation:
Salient_Object_Detection_Report.docx: A comprehensive technical report covering architecture design, 
loss functions (BCE + Dice), and quantitative analysis.  
SOD_Presentation.pptx: The final presentation deck summarizing the project flow, 
the pivot from CPU to GPU training, and live demo results.
