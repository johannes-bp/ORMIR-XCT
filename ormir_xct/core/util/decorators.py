"""
Created by:   Johannes Bereiter-Payr
Created on:   Sept 29, 2026

Description: Decorator functions to handle image type conversions for image processing functions.
"""

import functools
import copy

import SimpleITK as sitk
import numpy as np

from ormir_mids.utils import OMidsMedVolume, copy_metadata
from ormir_mids.utils.headers import remerge_headers, seperate_headers
from ormir_mids.utils.io import numpy_image_tuple


def _get_sitk_metadata(image):
    """
    Extract metadata from a SimpleITK image and return a deep copy.

    Parameters
    ----------
    image : SimpleITK.Image

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata of the image.
    """
    metadata = {}
    for key in image.GetMetaDataKeys():
        metadata[key] = copy.deepcopy(image.GetMetaData(key))
    return metadata


def _convert_to_sitk_image(image):
    """
    Convert an image to SimpleITK.Image if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    sitk_image : SimpleITK.Image
    """
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        im = sitk.GetImageFromArray(image[0])
        meta = image[1]
        for k, v in meta.items():
            im.SetMetaData(k, copy.deepcopy(v))
        return im
    elif isinstance(image, sitk.Image):
        return image
    elif isinstance(image, OMidsMedVolume):
        im = sitk.GetImageFromArray(image.data)
        meta = remerge_headers(image.omids_header, image.patient_header, image.extra_header)

        for k, v in meta.items():
            im.SetMetaData(k, copy.deepcopy(v))
        
        return im
    elif isinstance(image, np.ndarray):
        return sitk.GetImageFromArray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image, OMidsMedVolume, or numpy.ndarray.")


def _convert_to_omidsmedvolume(image):
    """
    Convert an image to OMidsMedVolume if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    omidsmedvolume : OMidsMedVolume
    """
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        omidsmed = OMidsMedVolume(image[0])
        omidsmed.omids_header, omidsmed.patient_header, omidsmed.extra_header = seperate_headers(image[1])
        return omidsmed
    elif isinstance(image, sitk.Image):
        omidsmed = OMidsMedVolume(sitk.GetArrayFromImage(image))
        meta = _get_sitk_metadata(image)
        omids_header, patient_header, extra_header = seperate_headers(meta)
        omidsmed.omids_header = omids_header
        omidsmed.patient_header = patient_header
        omidsmed.extra_header = extra_header
        return omidsmed
    elif isinstance(image, OMidsMedVolume):
        return image
    elif isinstance(image, np.ndarray):
        return OMidsMedVolume(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image.")


def _convert_to_numpy_array(image):
    """
    Convert an image to numpy.ndarray if it is not already. This function will return the raw data of the image, ignoring any metadata.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    numpy_array : numpy.ndarray
    """
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        return image[0]  # Already in the desired format
    elif isinstance(image, sitk.Image):
        return sitk.GetArrayFromImage(image)
    elif isinstance(image, OMidsMedVolume):
        return image.data
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image, OMidsMedVolume, or numpy.ndarray.")


def _convert_to_tuple(image):
    """
    Convert an image to a tuple of (numpy.ndarray, metadata_array) if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    tuple : (numpy.ndarray, metadata_array)
        A tuple containing the image data as a numpy array and the metadata.
    """
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        return image  # Already in the desired format
    elif isinstance(image, sitk.Image):
        return numpy_image_tuple(sitk.GetArrayFromImage(image), _get_sitk_metadata(image))
    elif isinstance(image, OMidsMedVolume):
        return numpy_image_tuple(image.data, remerge_headers(image.omids_header, image.patient_header, image.extra_header))
    elif isinstance(image, np.ndarray):
        return numpy_image_tuple(image, {})  # No metadata information available for raw numpy arrays
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image, OMidsMedVolume, or numpy.ndarray.")


def _convert_to_type(image, target_type):
    """
    Convert an image to the specified target type.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray
    target_type : type
        The desired output type. Can be SimpleITK.Image, OMidsMedVolume, tuple or numpy.ndarray.

    Returns
    -------
    converted_image : target_type
        The image converted to the specified target type.
    """
    if isinstance(image, target_type):
        return image  # Already in the desired format
    if target_type == tuple:
        return _convert_to_tuple(image)
    elif target_type == sitk.Image:
        return _convert_to_sitk_image(image)
    elif target_type == OMidsMedVolume:
        return _convert_to_omidsmedvolume(image)
    elif target_type == np.ndarray:
        return _convert_to_numpy_array(image)
    else:
        raise TypeError(f"Unsupported target type: {target_type}. Expected SimpleITK.Image, OMidsMedVolume, tuple, or numpy.ndarray.")


def request_sitk(_func=None, *image_arg_names, returns_image=True):
    """
    Decorator to convert any image type to SimpleITK.Image for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to SimpleITK.Image
            new_args = list(args)
            convert_to = None
            input_ref = None
            
            # Convert to list for easier manipulation
            image_arg_names = list(image_arg_names) if image_arg_names is not None else image_arg_names 

            # Default to the first argument if no names are provided
            if image_arg_names is None or len(image_arg_names) == 0:
                image_arg_names = [func.__code__.co_varnames[0]]  

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in image_arg_names:
                    convert_to = type(new_args[i])  # Store the original type for conversion back
                    
                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_sitk_image(kwargs[arg_name])
                    else:
                        new_args[i] = _convert_to_sitk_image(new_args[i])

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if returns_image:
                return _convert_to_type(result, convert_to)

            return result
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_numpy(_func=None, *image_arg_names, returns_image=True):
    """
    Decorator to convert any image type to numpy.ndarray for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to numpy.ndarray
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            image_arg_names = list(image_arg_names) if image_arg_names is not None else image_arg_names 

            # Default to the first argument if no names are provided
            if image_arg_names is None or len(image_arg_names) == 0:
                image_arg_names = [func.__code__.co_varnames[0]]  

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in image_arg_names:
                    convert_to = type(new_args[i])  # Store the original type for conversion back
                    
                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_numpy_array(kwargs[arg_name])
                    else:
                        new_args[i] = _convert_to_numpy_array(new_args[i])

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if returns_image:
                return _convert_to_type(result, convert_to)

            return result
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_omidsmedvolume(_func=None, *image_arg_names, returns_image=True):
    """
    Decorator to convert any image type to OMidsMedVolume for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to OMidsMedVolume
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            image_arg_names = list(image_arg_names) if image_arg_names is not None else image_arg_names 

            # Default to the first argument if no names are provided
            if image_arg_names is None or len(image_arg_names) == 0:
                image_arg_names = [func.__code__.co_varnames[0]]  

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in image_arg_names:
                    convert_to = type(new_args[i])  # Store the original type for conversion back

                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_omidsmedvolume(kwargs[arg_name])
                    else:
                        new_args[i] = _convert_to_omidsmedvolume(new_args[i])

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if returns_image:
                return _convert_to_type(result, convert_to)

            return result
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_tuple(_func=None, *image_arg_names, returns_image=True):
    """
    Decorator to convert any image type to a tuple of (numpy.ndarray, metadata_array) for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to tuple of (numpy.ndarray, metadata_array)
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            image_arg_names = list(image_arg_names) if image_arg_names is not None else image_arg_names 

            # Default to the first argument if no names are provided
            if image_arg_names is None or len(image_arg_names) == 0:
                image_arg_names = [func.__code__.co_varnames[0]]  

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in image_arg_names:
                    convert_to = type(new_args[i])  # Store the original type for conversion back

                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_tuple(kwargs[arg_name])
                    else:
                        new_args[i] = _convert_to_tuple(new_args[i])

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if returns_image:
                return _convert_to_type(result, convert_to)

            return result
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)
    