import cv2
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans



class ColorPrediction:
    def __init__(self):
        pass


    def find_main_colors(image, min_clusters, max_clusters, downsample_factor = 0.95, increase_elbow = 0, verbose=False, plot=False):
        """Function to find the optimal number of clusters using the elbow method.
        
        Parameters:
        ----------
        data: numpy array
            The data points to cluster.
        min_clusters: int
            The minimum number of clusters to consider.
        max_clusters: int
            The maximum number of clusters to consider.
        downsample_factor: float
            The factor to downsample the image by.
        increase_elbow: int
            The number of clusters to increase the elbow point by.
        verbose: bool
            Whether to print the optimal values.
        plot: bool
            Whether to plot the elbow plot and the optimal centroids.

        """

        def find_elbow_point(inertias, min_k, max_k):
            """ Function to find the elbow point in the inertia plot.

            Parameters:
            ----------
            inertias: list
                The list of inertia values for different k values.
            min_k: int  
                The minimum number of clusters to consider.
            max_k: int
                The maximum number of clusters to consider.
            
            """

            # Generate the range of k values
            K = range(min_k, max_k)

            kn = KneeLocator(K, inertias, curve='convex', direction='decreasing')
            return kn.elbow  

        def sort_lists_based_on_first(percentages, rgb_colors):
            """ Function to sort two lists based on the first list.
            
            Parameters:
            ----------
            percentages: list
                The list of percentages.
            rgb_colors: list
                The list of RGB colors.

            """

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
        
        def plot_rgb_squares_with_text(rgb_colors, text_labels, square_size=20):
            """ Function to plot RGB color squares with text labels (percentages).

            Parameters:
            ----------
            rgb_colors: list
                The list of RGB colors.
            text_labels: list
                The list of text labels, usually percentages.
            square_size: int
                The size of the square to plot.
            """

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
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))  # background
                

        # Function to create a 3D scatter plot of the RGB color space with centroids 
        def rgb_scatterplot(image, centroids = None):
            """ Function to create a 3D scatter plot of the RGB color space, optionally add centroids.

            Parameters:
            ----------
            image: numpy array
                The image to plot.
            downsample_factor: int
                The factor to downsample the image by.
            centroids: numpy array
                The centroids to plot.
            
            """

            # Reshape the image data
            if image.ndim == 3:
                h, w, c = image.shape
                image_data = image.reshape((h * w, c))

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

        def downsample_image(image, scale_factor):
            """ Downsample the given image by the specified scale factor.

            Parameters:
            ----------
            image: numpy array
                The image to downsample.
            scale_factor: float
                The scale factor to downsample the image by.

            """

            # Calculate the new dimensions
            new_width = int(image.shape[1] * (1 - scale_factor))
            new_height = int(image.shape[0] * (1 - scale_factor))
            new_dimensions = (new_width, new_height)

            # Downsample the image
            downsampled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
            return downsampled_image

        # Also look for clusters equal to max_clusters
        max_clusters  += 1

        # Convert the image to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        downsampled_image = downsample_image(image_rgb, downsample_factor)
        # Get the pixel color values as an array
        data = downsampled_image.reshape(-1, 3)

        # Initialize lists to store the inertias, percentages, and centroids
        inertias = []
        percentages = []
        centroids = []

        # Iterate over the range of k values
        K = range(min_clusters, max_clusters)
        for k in K:
            
            if verbose:
                print("Calculating for k = ", k)

            # Initialise the KMeans algorithm, fit and append the inertia values to the list
            kmeans = KMeans(n_clusters=k, algorithm='lloyd')
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

            # Calculate the percentage of points in each cluster
            _, counts = np.unique(kmeans.labels_, return_counts=True)
            centroids.append(kmeans.cluster_centers_)
            percentages.append(counts / len(data) * 100)
            
        # Plot the elbow plot
        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(K, inertias, 'bx-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal k')
        
        # Find the optimal k, centroids, and percentages
        optimal_k = find_elbow_point(inertias, min_clusters, max_clusters)
        if optimal_k is None:
            print("The elbow point could not be found. Try increasing the range of k values.")
            return None, None, None

        optimal_centroids = centroids[optimal_k - min_clusters + increase_elbow].astype(int)
        optimal_percentages = percentages[optimal_k - min_clusters + increase_elbow].astype(int)
        optimal_percentages, optimal_centroids = sort_lists_based_on_first(optimal_percentages, optimal_centroids)

        # Print the optimal values
        if verbose:
            print("The optimal number of clusters is: ", optimal_k + increase_elbow)
            print("The colors associated with each cluster are: \n", optimal_centroids)
            print("The percentage of points in each cluster are: ", optimal_percentages)
            
        # Plot the optimal centroid colors with percentages and a 3D scatter plot
        if plot:
            plot_rgb_squares_with_text(optimal_centroids, optimal_percentages, square_size=50)
            rgb_scatterplot(downsampled_image, optimal_centroids)
            plt.show()

        return optimal_centroids, optimal_k, optimal_percentages