import pylab
from pylab import plt as _plt
from pylab import get_current_fig_manager
matplotlib_backend=pylab.matplotlib.backends.backend


def axis_ij(g=None):
    if g is None:
        g = _plt.gca()
    bottom, top = g.get_ylim()  
    if top>bottom:
        g.set_ylim(top, bottom)
    else:
        pass
def axis_xy(g=None):
    if g is None:
        g = _plt.gca()
    bottom, top = g.get_ylim()  
    if top<bottom:
        g.set_ylim(top,bottom)
    else:
        pass    
    
def maximize_figure(fig=None):
    if fig is None:
        fig = _plt.gcf()
    
    mng = _plt.get_current_fig_manager()
    try:        
        mng.frame.Maximize(True)      
    except AttributeError:
        print "Failed to maximize figure."
        
        
        
def set_figure_size_and_location(x=50,y=50,width=400,height=400):           
    if matplotlib_backend in ['WX','WXAgg']:        
        thismanager = get_current_fig_manager()
        thismanager.window.SetPosition((x, y)) 
        thismanager.window.SetSize((width,height))               
    
    elif matplotlib_backend=='Qt4Agg':
        thismanager = get_current_fig_manager()
        thismanager.window.setGeometry(x,y,width,height)                
    else:
        raise NotImplementedError(matplotlib_backend)