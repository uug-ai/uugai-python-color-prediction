import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans



IMAGE_PATH = 'python_color_prediction/data/flowers2.jpeg'



# Function to find the optimal number of clusters using the elbow method
def elbow_method(data, max_clusters, verbose=True, plot=True):

    # Find optimal elbow, using the KneeLocator from the kneed library
    def find_elbow_point(inertias, max_k):
        # Generate the range of k values (minimum 1, maximum max_k)
        K = range(1, max_k + 1)
        kn = KneeLocator(K, inertias, curve='convex', direction='decreasing')
        return kn.elbow
    
    def sort_lists_based_on_first(percentages, rgb_colors):
        if len(percentages) != len(rgb_colors):
            raise ValueError("The length of percentages and rgb_colors must be the same.")
        
        # Convert percentages to a numpy array
        percentages_array = np.array(percentages)
        
        # Get the indices that would sort the percentages array
        sorted_indices = np.argsort(percentages_array)[::-1]
        
        # Use the sorted indices to sort both lists
        sorted_percentages = percentages_array[sorted_indices]
        sorted_rgb_colors = np.array(rgb_colors)[sorted_indices]
        
        return sorted_percentages, sorted_rgb_colors


    # Initialize lists to store the inertias, percentages, and centroids
    inertias = []
    percentages = []
    centroids = []

    # Iterate over the range of k values
    K = range(1, max_clusters + 1)
    for k in K:
        
        if verbose:
            print("Calculating for k = ", k)

        # Fit the KMeans model
        kmeans = KMeans(n_clusters=k, algorithm='lloyd')
        kmeans.fit(data)
        # Append the inertia value to the list
        inertias.append(kmeans.inertia_)

        # Count the number of points in each cluster
        _, counts = np.unique(kmeans.labels_, return_counts=True)
        centroids.append(kmeans.cluster_centers_)
        # Calculate percentages
        percentages.append(counts / len(data) * 100)
        
    # Plot the elbow plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
    
    # Find the optimal k, centroids, and percentages
    optimal_k = find_elbow_point(inertias, max_clusters)
    optimal_centroids = centroids[optimal_k - 1].astype(int)
    optimal_percentages = percentages[optimal_k - 1].astype(int)

    optimal_percentages, optimal_centroids = sort_lists_based_on_first(optimal_percentages, optimal_centroids)

    # Print the optimal values
    if verbose:
        print("The optimal number of clusters is: ", optimal_k)
        print("The colors associated with each cluster are: \n", optimal_centroids)
        print("The percentage of points in each cluster are: ", optimal_percentages)
        
    # Plot the optimal centroid colors with percentages
    if plot:
        plot_rgb_squares_with_text(optimal_centroids, optimal_percentages, square_size=50)

    return optimal_k, optimal_centroids, optimal_percentages





def plot_rgb_squares_with_text(rgb_colors, text_labels, square_size=20):
    if len(rgb_colors) != len(text_labels):
        raise ValueError("The length of rgb_colors and text_labels must be the same.")
    
    # Create an image array with the RGB colors
    num_colors = len(rgb_colors)
    image_array = np.zeros((square_size, square_size * num_colors, 3), dtype=np.uint8)
    
    for i, color in enumerate(rgb_colors):
        image_array[:, i*square_size:(i+1)*square_size, :] = color
    
    # Plot the image array
    fig, ax = plt.subplots(figsize=(num_colors, 2))
    ax.imshow(image_array)
    ax.axis('off')
    
    # Add text labels
    for i, text in enumerate(text_labels):
        ax.text(
            (i + 0.5) * square_size,   # x position
            square_size / 2,           # y position
            str(text)+'%',                      # text
            color='black',             # text color
            fontsize=12,               # text size
            ha='center',               # horizontal alignment
            va='center',               # vertical alignment
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')  # background
        )

    


# Function to create a 3D scatter plot of the RGB color space with centroids 
def plot_rgb_3d_with_centroids(image_path, downsample_factor, centroids = None):

    # Function to load and downsample the image
    def load_and_downsample_image(image_path, downsample_factor):
        # Load the image
        image = Image.open(image_path)
        # Calculate new size
        new_size = (image.width // downsample_factor, image.height // downsample_factor)
        # Downsample the image
        downsampled_image = image.resize(new_size, Image.LANCZOS)
        # Convert to numpy array
        image_data = np.array(downsampled_image)
        return image_data
    
    # Load and downsample the image
    image_data = load_and_downsample_image(image_path, downsample_factor)

    # Reshape the image data
    if image_data.ndim == 3:
        h, w, c = image_data.shape
        image_data = image_data.reshape((h * w, c))
    
    # Normalize the image data
    image_data = image_data / 255.0
    
    # Create a matplotlib figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot for the image data
    ax.scatter(image_data[:, 0], image_data[:, 1], image_data[:, 2], c=image_data, marker='o')
    
    # Plot centroids if provided
    if centroids is not None:
        # Normalize the centroids
        centroids = centroids / 255.0
        # Plot centroids as transparent spheres, in color representing the centroid
        for centroid in centroids:
            ax.scatter(centroid[0], centroid[1], centroid[2], color=centroid, s=2000, alpha=0.5, edgecolors='w', linewidths=2)
    
    # Set the axis labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Color Space with Centroids')
    


if __name__ == '__main__':
    # Read the image
    image = cv2.imread(IMAGE_PATH)

    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Get the pixel color values as an array
    data = image_rgb.reshape(-1, 3)

    # Call the elbow_method function
    optimal_k, optimal_centroids, optimal_percentages = elbow_method(data, max_clusters=10)

    # Plot the 3D RGB color space with centroids
    plot_rgb_3d_with_centroids(IMAGE_PATH, 30, optimal_centroids)

    plt.show()
