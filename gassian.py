import streamlit as st
import cv2 as cv
import numpy as np
import joblib
import tempfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your trained GMM model
gmm_model = joblib.load("gmm_model.pkl")


def preprocess_frame(frame):
    # Resize
    resized_frame = cv.resize(frame, (640, 240))
    # Gaussian blur
    blurred_frame = cv.GaussianBlur(resized_frame, (5, 5), 0)
    # Sharpening filter
    sharpened_frame = cv.addWeighted(resized_frame, 1.5, blurred_frame, -0.5, 0)

    return sharpened_frame

def extract_features_from_frame(prev_gray, gray, mask):

    # cv.imshow("frame",gray)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # cv.imshow("mask",rgb)
    feature_vector = np.concatenate((magnitude.flatten(), angle.flatten()))
    return feature_vector

def main():
    st.title("Fall Detection App")

    # Upload video file or provide video URL
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_video is not None:
        if st.button("Process Video"):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())

            cap = cv.VideoCapture(temp_file.name)

            ret, first_frame = cap.read()
            if not ret:
                st.error("Error reading the first frame")
                return
            mask = np.zeros_like(first_frame)
            prev_gray = preprocess_frame(cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY))

            features_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = preprocess_frame(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
                feature_vector = extract_features_from_frame(prev_gray, gray, mask)
                features_list.append(feature_vector)
                prev_gray = gray

            cap.release()

            # Now you have features_list containing features of each frame
            st.success("Feature extraction completed!")
            st.write("Shape of features_list:", np.array(features_list).shape)
            features_array = np.array(features_list)
            n_components_pca = 90
            pca = PCA(n_components=n_components_pca)

            X_pca = pca.fit_transform(features_array)
            cluster_labels = gmm_model.predict(X_pca)

            # Visualize clustering results in a 2D scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
            plt.title('Clustering Results')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster')
            st.pyplot(plt)
            c1_array=[]
            c2_array=[]
            fall_frames=[]
            train_cluster_labels = gmm_model.predict(X_pca)

            cluster_frames = {}
            for cluster_label in np.unique(train_cluster_labels):
                cluster_frames[cluster_label] = []

            for i, label in enumerate(train_cluster_labels):
                cluster_frames[label].append(i)
            for cluster_label, frames in cluster_frames.items():
                print(f"Cluster {cluster_label}: {len(frames)} frames")
                if cluster_label == 0:  # cluster 0 corresponds to the "not fall" class
                    c1_array = frames
                elif cluster_label == 1:  # cluster 1 corresponds to the "fall" class
                    c2_array = frames
            if len(c1_array)<len(c2_array):
                fall_frames = c1_array
            else:
                fall_frames = c2_array
            
            st.write("Fall detected")
            st.write(fall_frames)
            
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv.CAP_PROP_FPS)
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            video_writer = cv.VideoWriter('path_to_generated_video.mp4', fourcc, fps, (frame_width, frame_height))
            
            
            frame_counter=0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_counter+=1
                if frame_counter in fall_frames:
                    video_writer.write(frame)

            cap.release()
            video_writer.release()
            cv.destroyAllWindows()
            st.write("Fall detected")



if __name__ == "__main__":

    main()
