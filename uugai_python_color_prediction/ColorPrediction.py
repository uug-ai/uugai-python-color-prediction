import cv2
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



class ColorPrediction:
    def __init__(self):
        pass


    def find_main_colors(image, coding : str = 'BGR', min_clusters : int = 1, max_clusters : int = 8, downsample_factor : float = 0, increase_elbow : int = 0, verbose : bool = False, plot : bool = False):
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

        # Downsample the image if the downsample factor is not 0
        if downsample_factor != 0:
            image = ColorPrediction.downsample_image(image=image, scale_factor=downsample_factor)
        else:
            image = image


        # Depending on the coding, create the data and image arrays.
        image, data = ColorPrediction.create_data_and_image(
            in_image=image,
            coding=coding)


        # Calculate the inertia values, percentages, and centroids with the KMeans algorithm
        kmeans_data = ColorPrediction.kmeans_inertias(
            data=data,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            verbose = verbose)


        # Find the optimal k-clusters value using the elbow method
        optimal_k = ColorPrediction.find_elbow_point(
            inertias=[kmeans_data[k]['inertias'] for k in kmeans_data.keys()],
            min_clusters=min_clusters,
            max_clusters=max_clusters)


        # Print the optimal values
        if verbose:
            if optimal_k is not None:
                print(f"\nThe optimal number of clusters is: {optimal_k}")
                for color, percentage in zip(kmeans_data[optimal_k]['centroids'], kmeans_data[optimal_k]['percentages']):
                    print(f"Color: {color}, Percentage: {percentage}%")

            
                if increase_elbow != 0:
                    print(f"\nIncreasing the elbow point by {increase_elbow} cluster(s):")
                    print(f"New number of clusters is: {optimal_k + increase_elbow}")
                    for color, percentage in zip(kmeans_data[optimal_k + increase_elbow]['centroids'], kmeans_data[optimal_k + increase_elbow]['percentages']):
                        print(f"Color: {color}, Percentage: {percentage}%")
            else:
                print("No elbow point found. Please try again with different parameters.")
        

        # Plot the inertia values, RGB squares, and RGB scatterplot
        if plot and optimal_k is not None:
            ColorPrediction.plot_kmeans_inertias(kmeans_data)
            ColorPrediction.plot_bgr_squares(kmeans_data, optimal_k + increase_elbow)
            ColorPrediction.plot_bgr_scatter(data, kmeans_data, optimal_k + increase_elbow)
            plt.show()
            
        return optimal_k, kmeans_data
    

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


    def create_data_and_image(in_image, coding):
        """ Function to create the data and image arrays based on the input image and coding.

        Parameters:
        ----------
        in_image: numpy array
            The input image.
        coding: str
            The coding of the image, either 'BGR', 'RGB', 'BGRA', or 'RGBA'.

        """

        # If the image is in BGR format, just return the image and data back
        if coding == 'BGR':
            image = in_image
            data = image.reshape(-1, 3)
            return image, data
        
        # If the image is in RGB format, convert it to BGR and return the image and data back
        elif coding == 'RGB':
            image = cv2.cvtColor(in_image, cv2.COLOR_RGB2BGR)
            data = image.reshape(-1, 3)
            return image, data
        
        # If the image is in BGRA format, convert it to BGR and return the image.
        # Also, remove the transparant data points from the data array.
        elif coding == 'BGRA':
            image = cv2.cvtColor(in_image, cv2.COLOR_BGRA2BGR)

            data = in_image.reshape(-1, 4)
            non_transparant_data = data[data[:, 3] != 0]
            bgr_data = non_transparant_data[:, :3]

            return image, bgr_data
        
        # If the image is in RGBA format, convert it to BGR and return the image.
        # Also, remove the transparant data points from the data array, and convert the RGB data to BGR.
        elif coding == 'RGBA':
            image = cv2.cvtColor(in_image, cv2.COLOR_RGBA2BGR)

            data = in_image.reshape(-1, 4)
            non_transparant_data = data[data[:, 3] != 0]
            rgb_data = non_transparant_data[:, :3]
            bgr_data = rgb_data[:, [2, 1, 0]]

            return image, bgr_data

        # If the coding is invalid, raise a ValueError
        else:
            return ValueError("Invalid coding. Please choose one of the following: 'BGR', 'RGB', 'BGRA', 'RGBA'")        
        

    def kmeans_inertias(data, min_clusters, max_clusters, verbose = False):
        """ Function to calculate the KMeans-inertia values for different k values.
        
        Parameters:
        ----------
        data: numpy array
            The data points to cluster.
        min_clusters: int
            The minimum number of clusters to consider.
        max_clusters: int
            The maximum number of clusters to consider.
        
        """

        # Initialize lists to store the inertias, percentages, and centroids

        kmeans_data = {}

        # Iterate over the range of k values, +1 to include max_clusters
        for k in range(min_clusters, max_clusters + 1):

            if verbose:
                print(f"Calculating KMeans for k={k}...")

            kclusters_data = {
            "inertias": [],
            "centroids": [],
            "percentages": []
            }

            # Initialise the KMeans algorithm and fit the data
            kmeans = KMeans(n_clusters=k, algorithm='lloyd')
            kmeans.fit(data)

            # Store the inertia, centroids, and percentages in the lists for each k value
            kclusters_data['inertias'] = kmeans.inertia_
            kclusters_data['centroids'] = kmeans.cluster_centers_.astype(int)
            _, counts = np.unique(kmeans.labels_, return_counts=True)
            kclusters_data['percentages'] = (counts / len(data) * 100).round(1)

            # Store the data in the kmeans_data dictionary
            kmeans_data[k] = kclusters_data

        return kmeans_data


    def find_elbow_point(inertias, min_clusters, max_clusters):
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

        # Use the KneeLocator to find the elbow point
        kn = KneeLocator(
            x = range(min_clusters, max_clusters + 1),
            y = inertias,
            S=2.0,
            curve='convex', 
            direction='decreasing')
        
        # Return the elbow point, if no elbow point is found, return None
        return kn.elbow
    

    def plot_kmeans_inertias(kmeans_data):
        """ Function to plot the KMeans inertia values for different k values.

        Parameters:
        ----------
        kmeans_data: dict
            The dictionary containing the KMeans data.
        
        """

        plt.figure(figsize=(8, 4))
        plt.plot(
            kmeans_data.keys(),
            [kmeans_data[k]["inertias"] for k in kmeans_data.keys()],
            'bx-')
        
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Inertias from KMeans-clustering technique used in the elbow method to find the optimal k-value')


    def plot_bgr_squares(kmeans_data, k, square_size=20):
        """ Function to plot RGB color squares with text labels (percentages).

        Parameters:
        ----------
        kmeans_data: dict
            The dictionary containing the KMeans data.
        k: int
            The number of clusters to plot.
        square_size: int
            The size of the square to plot.

        """
        
        # Create an image array with the BGR colors
        image_array = np.zeros((square_size, square_size * k, 3), dtype=np.uint8)
        
        for i, color in enumerate(kmeans_data[k]['centroids']):
            image_array[:, i*square_size:(i+1)*square_size, :] = color
        
        # Plot the image array
        fig, ax = plt.subplots(figsize=(k, 2))
        ax.imshow(image_array)
        ax.axis('off')
        
        # Add text labels
        for i, text in enumerate(kmeans_data[k]['percentages']):
            ax.text(
                (i + 0.5) * square_size,                                            # x position
                square_size / 2,                                                    # y position
                str(text)+'%',                                                      # text
                color='black',                                                      # text color
                fontsize=12,                                                        # text size
                ha='center',                                                        # horizontal alignment
                va='center',                                                        # vertical alignment
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))  # background
                

    # Function to create a 3D scatter plot of the RGB color space with centroids 
    def plot_bgr_scatter(image_data, kmeans_data, optimal_k):
        """ Function to create a 3D scatter plot of the RGB color space, optionally add centroids.

        Parameters:
        ----------
        image_data: numpy array
            The image data to plot.
        kmeans_data: dict
            The dictionary containing the KMeans data.
        optimal_k: int
            The optimal number of clusters to plot.
        
        """

        # Create a matplotlib figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Normalise the image data and plot the scatter plot, these are the pixels in the image
        normalised_image_data = image_data / 255.0
        ax.scatter(normalised_image_data[:, 0], normalised_image_data[:, 1], normalised_image_data[:, 2], c=normalised_image_data, marker='o')

        # Normalise the centroids and plot the centroids, these are the main colors in the image
        normalised_centroids = kmeans_data[optimal_k]['centroids']/255
        ax.scatter(normalised_centroids[:, 0], normalised_centroids[:, 1], normalised_centroids[:, 2], color=normalised_centroids, s=2000, alpha=0.5, edgecolors='w', linewidths=2)
                
        # Set the axis labels and title
        ax.set_xlabel('Blue')
        ax.set_ylabel('Green')
        ax.set_zlabel('Red')
        ax.set_title('3D BGR color-space with centroids')
