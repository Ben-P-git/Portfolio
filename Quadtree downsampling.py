# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:05:51 2025

@author: BenPa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:59:06 2025

@author: BenPa
"""

def tree_to_mat(bounds,displacements,mask):
    """
    Creates r2 array from quadtree form    
    
    Parameters
    ----------
 
    bounds : TYPE - A list of tuples 
        DESCRIPTION - List of bounds corresponding to the displacements
    displacements : TYPE  - List   
     DESCRIPTION - A list of downsampled displacements 
    Mask: TYPE: array 
    DESCRIPTION: mask of the corresponding array of displacements 

    Returns
    -------
    Array: upsampled array
    
        

    """
    import numpy as np
    import numpy.ma as ma
    
    array = np.zeros(mask.shape)  # predefine r2 array
    

    for i,j in enumerate(displacements):
        (y_start, x_start) , (y_end, x_end )= bounds[i] # extract bounds
        array[y_start:y_end,x_start:x_end] = j #input array
    
    array =ma.array(array,mask=mask) # redefine array
    
    return array


#%%

def plot_downsampled(downsampled,original,bounds,mask,png_path = './'):
    """
    Generates a comparison of the original data and the downsampled version.

    Parameters
    ----------
    downsampled : TYPE - r2 array
        DESCRIPTION - List of the cummulative interferogram 
    original : TYPE - r2 array of the compressed original data 
        DESCRIPTION.
    bounds : TYPE - list of tuples
        DESCRIPTION - Bounds for the quadtree downsampled data
    mask : TYPE - array
        DESCRIPTION - mask for the data 
    png_path : TYPE - The original data , optional
        DESCRIPTION - path to save the figure The default is './'.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    from licsalert.aux import col_to_ma
    import numpy as np
    import pdb
    
    
    m = downsampled.shape[0] # define length of timeseries
    refs = [0,int(m/3),int((2*m)/3),-1]  # define the references for each ifg between 6 inc start and end
    
    cum = np.zeros(downsampled.shape) # initiate array for the cumulative downsampled data
    cumulative = np.zeros(original.shape) # initiate array for cumulative original data
    
    
    for inc in range(m):
        cum[inc,:] = cum[inc-1,:] + downsampled[inc,:] # generate cumulative downsampled array
        cumulative[inc,:] = cumulative[inc-1,:]+original[inc,:] # generate cumulative non downsampled array
    
    fig, axs = plt.subplots(4,4,figsize=(40,40)) # predefine figure
    
    for i,j in enumerate(refs):
        
        
        # upsample the array to original size
        cum_temp = col_to_ma(cumulative[j,:], mask) 
        cum_temp_downsampled = tree_to_mat(bounds, cum[j,:], mask)
        inc_temp = col_to_ma(original[j,:], mask)
        inc_temp_downsampled = tree_to_mat(bounds, downsampled[j,:], mask)
        
        #calculate the norm
        inc_norm= np.linalg.norm(inc_temp-inc_temp_downsampled,ord=2)
        cum_norm= np.linalg.norm(cum_temp-cum_temp_downsampled,ord=2)/j # divide by number of interferogram to account for scaling of ifgs
        
        
        #plot the data and add norms to the title 
        axs[i,0].imshow(inc_temp)
        axs[i,0].set_title(f"frobenius norm of {inc_norm}")
        axs[i,1].imshow(inc_temp_downsampled)
        axs[i,2].imshow(cum_temp)
        axs[i,2].set_title(f"frobenius norm of {cum_norm}")
        axs[i,3].imshow(cum_temp_downsampled)
    # save fig    
    plt.savefig(f"quadtree_plot.png")
    plt.show()
    return 
    
    
    
    
    
    
    





#%%

def quadtree_downsample(displacements,mask,param,min_variance=1e-4, quadtree_depth=9):
    """
    Downsampling function that takes a series of incremental interferograms and downsamples them according to the variance within each image
    using a quadtreee based downsampling technique.

    Parameters
    ----------
    displacements : TYPE - Displacement dictionary 
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    param : TYPE
        DESCRIPTION.
    min_variance : TYPE, optional
        DESCRIPTION. The default is 1e-4.
    quadtree_depth : TYPE, optional
        DESCRIPTION. The default is 9.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # function taken from https://github.com/matthew-gaddes/LiCSAlert/tree/master/licsalert
    from licsalert.aux import col_to_ma
    import numpy as np 
    import pdb
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    
    
    #define the quedtree node as a class 
    class QuadtreeNode:
        def __init__(self, bounds, displacements,dem, parent_displacement_value,param):
            self.bounds = bounds
            self.children = []
    
            # Check if displacements contain only masked values
            if displacements.size == 0 or np.ma.count(displacements) == 0:
                self.displacement_value = None  # Mark as invalid
                self.variance_value = None
            else:
                self.displacement_value = np.ma.mean(displacements)
                self.variance_value = (param[0]*np.ma.std(displacements))+(param[1]*np.ma.std(dem))
    
            self.midpoint = (bounds[1][0] + bounds[0][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2
    
        def is_leaf(self):
            return len(self.children) == 0
    
        def is_empty(self):
            """ Check if the node contains only masked values """
            return self.displacement_value is None
    
    
    
    # Compute child bounds for each quadrant
    def compute_child_bounds(parent_bounds, child_index):
        (y_min,x_min), (y_max,x_max) = parent_bounds
        x_mid, y_mid = int( (x_min + x_max) / 2 ),  int ( (y_min + y_max) / 2 )
        child_bounds = [
            ((y_min, x_min), (y_mid, x_mid)),  # Bottom-left
            ((y_mid, x_min), (y_max, x_mid)),  # Bottom-right
            ((y_min, x_mid), (y_mid, x_max)),  # Top-left
            ((y_mid, x_mid), (y_max, x_max)),  # Top-right
        ]
        return child_bounds[child_index]
    

    
    
    # Recursively subdivide the quadtree
    def subdivide(node,displacement,dem, max_depth, min_variance,depth=0):
        if depth >= max_depth or node.is_empty() or min_variance > node.variance_value:
            print(depth)
            return  # Stop if max depth is reached or node is empty
        for i in range(4):
            child_bounds = compute_child_bounds(node.bounds, i)
            (y_start, x_start),( y_end, x_end )= child_bounds
            child_displacements = displacement[y_start:y_end,x_start:x_end]
            child_dem = dem[y_start:y_end,x_start:x_end]
            child = QuadtreeNode(child_bounds, child_displacements,child_dem, node.displacement_value,param)
    
            if not child.is_empty():# **Only add non-empty children**
                node.children.append(child)
                subdivide(child, displacement,dem,max_depth, min_variance,depth + 1)
    
    
    def query_quadtree(node, resolution):
        if node.is_empty() or not node.children or resolution == 0:
            return [node.midpoint], [node.displacement_value], [node.bounds] if not node.is_empty() else ([], [], [])
    
        results = []
        midpoints = []
        bounds = []
        for child in node.children:
            midpoint, result, bound = query_quadtree(child, resolution - 1)
            results.extend(result)
            midpoints.extend(midpoint)
            bounds.extend(bound)
        
        return midpoints, results, bounds

    def ts_downsampling(ts, root_bounds, bounds,mask):
        """
        Downsamples time series data based on quadtree segmentation.
        
        Ensures no NaNs while excluding masked values.
        """
        
        downsampled_array = np.zeros((len(ts[:,0]), len(bounds)))  # Initialize with zeros
        for i in range(ts.shape[0]):  # Iterate over time series frames
            print("percent complete:", i*100/ts.shape[0])
            temp = col_to_ma(ts[i,:],mask)

            for j, bound in enumerate(bounds):
                (y_start, x_start),( y_end, x_end )= bound
                # Ensure valid indexing
                if x_end <= x_start or y_end <= y_start:
                    print("hi")
                    downsampled_array[i, j] = 0  # Assign zero if invalid
                    continue
                region = temp[y_start:y_end, x_start:x_end]
                mask_temp = mask[y_start:y_end, x_start:x_end]
                # Compute mean only from unmasked values
                valid_values = region[~mask_temp]
                
                if valid_values.size > 0:
                    downsampled_array[i, j] = np.mean(valid_values)  # Use mean of valid values
                else:
                    downsampled_array[i, j] = 0  # Assign zero if everything is masked
        return downsampled_array
    
    
    def dem_downsampling(dem,  bounds,mask):
        downsampled_array = np.zeros(len(bounds))  # Initialize with zeros
        
          # Iterate over time series frames
        for j, bound in enumerate(bounds):
            (y_start, x_start),( y_end, x_end )= bound
            # Ensure valid indexing
            if x_end <= x_start or y_end <= y_start:
                print("hi")
                downsampled_array[j] = 0  # Assign zero if invalid
                continue
            
            region = dem[y_start:y_end, x_start:x_end]
            mask_temp = mask[y_start:y_end, x_start:x_end]
            
            # Compute mean only from unmasked values
            valid_values = region[~mask_temp]
            valid_values= np.nan_to_num(valid_values)
            if valid_values.size > 0:
                downsampled_array[j] = np.mean(valid_values)  # Use mean of valid values
            else:
                downsampled_array[j] = 0  # Assign zero if everything is masked
        return downsampled_array
    
    
    
    #define the initial cumulative array and rescale it between 0 and 1 n
    cum= np.sum(displacements["incremental"],axis=0) # import 2d array
    cum_rescale = (cum-np.min(cum))/(np.max(cum)-np.min(cum))
    #
    cum_rescale = col_to_ma(cum_rescale,mask)

    dem= ma.compressed(ma.array(displacements["dem"],mask=mask))
    dem = np.nan_to_num(dem)
    if np.max(dem)==np.min(dem):
        dem_rescale= dem

        print("dem has no range to be rescaled")
    else:
        dem_rescale = (dem-np.min(dem))/(np.max(dem)-np.min(dem))

    dem_rescale = col_to_ma(dem_rescale,mask)
    
    
    root_bounds = ((0,0),(displacements["dem"].shape[0], displacements["dem"].shape[1]))

    
    incremental = col_to_ma(displacements["incremental"],np.stack([mask]*len(displacements["incremental"][:,0]),axis=2))
    # implement quadtree downsampling 
    
    quadtree = QuadtreeNode(root_bounds,cum_rescale,dem_rescale,np.mean(cum_rescale),param)#set it as the final cummulative displacement
    subdivide(quadtree,cum_rescale,dem_rescale, quadtree_depth,min_variance) #subdivide
    midpoints,results,bounds = query_quadtree(quadtree, quadtree_depth) # query to extract the resoltion
    
    print(f"Quadtree downsampled to : {len(results)} pixels.")
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(cum_rescale)
    axs[1].imshow(tree_to_mat(bounds, results, mask))
    fig.colorbar(None)
    plt.show()
    
    
    dem = dem_downsampling(displacements["dem"],bounds,mask)
    dem=tree_to_mat(bounds, dem, mask)
    downsampled = ts_downsampling(displacements["incremental"],root_bounds,bounds,mask)
    
    plot_downsampled(downsampled,displacements["incremental"],bounds,mask)
    plt.figure(figsize=(100,100))
    midpoints_arr = np.stack(midpoints)
    for i,j in enumerate(bounds):
        plt.plot([j[0][1],j[1][1],j[1][1],j[0][1]],[j[0][0],j[0][0],j[1][0],j[1][0]])
    plt.scatter(midpoints_arr[:,1],midpoints_arr[:,0])
    plt.savefig("DSAF.png")
    
    
    
    return downsampled,bounds,dem
    





