import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def regrid(df, nx_grid=10, ny_grid=10, gridding='linear'):
  """
  Regridding data
  """
  from scipy import interpolate

  xd, yd, zd = df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]
  xmin, xmax = xd.min(), xd.max()
  ymin, ymax = yd.min(), yd.max()  

  x = np.linspace(xmin, xmax, nx_grid)
  y = np.linspace(ymin, ymax, ny_grid)
  xi, yi = np.meshgrid(x, y)
  zi = interpolate.griddata((xd, yd), zd, (xi,yi), method=gridding)
  return xi, yi, zi

def pixelplot(df, nx_grid=10, ny_grid=10, cmap='plasma', 
              vmin=None, vmax=None, gridding='linear', interpolation=None):
  """
  Pixel plot from unstructured 3D data 
  
  NOTE: Does not honor discontinuities in data (e.g. fault unconformities)
  """  
  # Regrid data
  xi, yi, zi = regrid(df, nx_grid, ny_grid, gridding=gridding)

  # Extents
  xd, yd, zd = df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]
  xmin, xmax = xd.min(), xd.max()
  ymin, ymax = yd.min(), yd.max()    

  # Plot map
  plt.imshow(zi, origin='lower', extent=(xmin,xmax,ymin,ymax), cmap=cmap,
             vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation)
  xlabel, ylabel, zlabel = df.columns
  plt.xlabel(xlabel, size=15)
  plt.ylabel(ylabel, size=15)
  plt.title(zlabel, size=20, pad=10)
  plt.colorbar()  
  plt.xlim(xd.min(), xd.max())
  plt.ylim(yd.min(), yd.max())

def correlation(primary_df, secondary_df, method='cubic'):
  """
  Correlation between primary and secondary data
  """
  from sklearn.metrics import r2_score
  from scipy.stats import pearsonr
  from scipy.optimize import curve_fit
  from scipy import interpolate

  # Points (x,y) from secondary data
  points = secondary_df.iloc[:,:2].values

  # Data values from secondary data
  values = secondary_df.iloc[:,-1].values

  # Points (x,y) from primary data
  newpoints = primary_df.iloc[:,:2].values

  # Data values from primary data
  newvalues = primary_df.iloc[:,-1].values

  # Interpolate secondary data on points from primary data
  values_interp = interpolate.griddata(points, values, newpoints, method=method)

  # Linear regression
  def linear(x, a, b): return a*x+b
  [a,b], pcov = curve_fit(linear, newvalues, values_interp)
  xfit = np.sort(newvalues)
  yfit = linear(xfit, a, b)

  # Calculate Pearson correlation
  corr = pearsonr(newvalues, values_interp)

  # Plot
  plt.scatter(newvalues, values_interp, alpha=0.7)
  plt.plot(xfit, yfit, 'r', label='Corr: {:.3f}'.format(corr[0]))
  plt.xlabel("Primary Data", size=15)
  plt.ylabel("Secondary Data", size=15)
  plt.title("Primary-Secondary Crossplot", size=20, pad=10)
  plt.legend(fontsize=12)
  plt.show()
  
  return values_interp
