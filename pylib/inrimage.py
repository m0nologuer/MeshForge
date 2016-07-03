from os import path
import numpy as np
from struct import calcsize,pack,unpack
import gzip
from cStringIO import StringIO

specific_header_keys = ("XDIM","YDIM","ZDIM",
                        "VDIM","TYPE","PIXSIZE",
                        "SCALE","CPU",
                        "VX","VY","VZ",
                        "TX","TY","TZ")

def open_inrifile (filename) :
    """Open an inrimage file
    
    Manage the gz attribute
    """
    if path.splitext(filename)[1] in (".gz",".zip") :
        fzip = gzip.open(filename,'rb')
        f = StringIO(fzip.read())
        fzip.close()
    else :
        f = open(filename,'rb')
    
    return f

def _read_header (f) :
    """
    extract header from a stream and return it
    as a python dict
    """
    #read header string
    header = ""
    while header[-4:] != "##}\n" :
        header += f.read(256)
    
    #read infos in header
    prop = {}
    hstart = header.find("{\n") + 1
    hend = header.find("##}")
    infos = [gr for gr in header[hstart:hend].split("\n") \
                       if len(gr) > 0]
    
    #format infos
    for prop_def in infos :
        key,val = prop_def.split("=")
        prop[key] = val
    
    #return
    return prop

def read_inriheader (filename) :
    """Read only the header of an inrimage
    """
    f = open_inrifile(filename)
    prop = _read_header(f)
    f.close()
    return prop

def read_inrimage (filename) :
    """
    read an inrimage, either zipped or not according to extension
    """
    f = open_inrifile(filename)
    
    #read header
    prop = _read_header(f)
    
    #extract usefull infos to read image
    xdim = int(prop["XDIM"])
    ydim = int(prop["YDIM"])
    zdim = int(prop["ZDIM"])
    vdim = int(prop["VDIM"])
    if vdim != 1 :
        raise NotImplementedError("don't know how to handle vectorial pixel values")
    
    #find data type
    if prop["TYPE"] == "unsigned fixed" :
        try :
            pixsize = int(prop["PIXSIZE"].split(" ")[0])
            try :
                ntyp = eval("np.dtype(np.uint%d)" % pixsize)
            except AttributeError :
                raise UserWarning("undefined pix size: %s" % prop["PIXSIZE"])
        except KeyError :
            ntyp = np.dtype(np.int)
    elif prop["TYPE"] == "float" :
        try :
            pixsize = int(prop["PIXSIZE"].split(" ")[0])
            try :
                ntyp = eval("np.dtype(np.float%d)" % pixsize)
            except AttributeError :
                raise UserWarning("undefined pix size: %s" % prop["PIXSIZE"])
        except KeyError :
            ntyp = np.dtype(np.float)
    else :
        raise UserWarning("unable to read that type of datas : %s" % prop["TYPE"])
    
    #read datas
    size = ntyp.itemsize * xdim * ydim * zdim
    mat = np.fromstring(f.read(size),ntyp)
    mat = mat.reshape( (zdim,ydim,xdim) ).transpose()
    #return
    f.close()
    return mat,prop

def write_inrimage (mat, info, filename) :
    """
    write an inrimage zipped or not according to the extension
    """
    zipped = ( path.splitext(filename)[1] in (".gz",".zip") )
    
    if zipped :
        f = StringIO()
    else :
        f = open(filename,'wb')
    #im dimensions
    xdim,ydim,zdim = mat.shape
    info["XDIM"] = "%d" % xdim
    info["YDIM"] = "%d" % ydim
    info["ZDIM"] = "%d" % zdim
    info["VDIM"] = "1"
    #data type
    if mat.dtype == np.uint8 :
        info["TYPE"] = "unsigned fixed"
        info["PIXSIZE"] = "8 bits"
    elif mat.dtype == np.uint16 :
        info["TYPE"] = "unsigned fixed"
        info["PIXSIZE"] = "16 bits"
    elif mat.dtype == np.uint32 :
        info["TYPE"] = "unsigned fixed"
        info["PIXSIZE"] = "32 bits"
    elif mat.dtype == np.uint64 :
        info["TYPE"] = "unsigned fixed"
        info["PIXSIZE"] = "64 bits"
    elif mat.dtype == np.float32 :
        info["TYPE"] = "float"
        info["PIXSIZE"] = "32 bits"
    elif mat.dtype == np.float64 :
        info["TYPE"] = "float"
        info["PIXSIZE"] = "64 bits"
    #elif mat.dtype == np.float128 :
    #    info["TYPE"] = "float"
    #    info["PIXSIZE"] = "128 bits"
    else :
        raise UserWarning("unable to write that type of datas : %s" % str(mat.dtype) )
    #write header
    header = "#INRIMAGE-4#{\n"
    for k in specific_header_keys :#HACK pas bo to ensure order of specific headers
        header += "%s=%s\n" % (k,info[k])
    for k in set(info) - set(specific_header_keys) :
        header += "%s=%s\n" % (k,info[k])

    #fill header to be a multiple of 256
    header_size = len(header) + 4
    if (header_size % 256) > 0 :
        header += "\n" * ( 256 - header_size % 256 )
    header += "##}\n"
    f.write(header)
    
    #write datas
    f.write(mat.transpose().tostring() )
    
    #return
    if zipped :
        fzip = gzip.open(filename,'wb')
        fzip.write(f.getvalue() )
        fzip.close()
    f.close()

