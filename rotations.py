import numpy as np
import cPickle as pkl

class Quaternion:
    """Quaternions for 3D rotations"""
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        
    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternion from unit vector v and rotation angle theta
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))

        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.x[:, None] * other.x

        return self.__class__([(prod[0, 0] - prod[1, 1]
                                 - prod[2, 2] - prod[3, 3]),
                                (prod[0, 1] + prod[1, 0]
                                 + prod[2, 3] - prod[3, 2]),
                                (prod[0, 2] - prod[1, 3]
                                 + prod[2, 0] + prod[3, 1]),
                                (prod[0, 3] + prod[1, 2]
                                 - prod[2, 1] + prod[3, 0])])

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        # compute theta
        norm = np.sqrt((self.x ** 2).sum(0))
        theta = 2 * np.arccos(self.x[0] / norm)

        # compute the unit vector
        v = np.array(self.x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()
        c = np.cos(theta)
        s = np.sin(theta)

        return np.array([[v[0] * v[0] * (1. - c) + c,
                          v[0] * v[1] * (1. - c) - v[2] * s,
                          v[0] * v[2] * (1. - c) + v[1] * s],
                         [v[1] * v[0] * (1. - c) + v[2] * s,
                          v[1] * v[1] * (1. - c) + c,
                          v[1] * v[2] * (1. - c) - v[0] * s],
                         [v[2] * v[0] * (1. - c) - v[1] * s,
                          v[2] * v[1] * (1. - c) + v[0] * s,
                          v[2] * v[2] * (1. - c) + c]])
    
    
    class SkelAxes(plt.Axes):
        # base bone is perpendicular to z at z=+1 - this probably isn't right for our case
        one_bone = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]])

        def __init__(self, fig, rect=[0, 0, 1, 1], *args, **kwargs):
            kwargs.update(dict(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), frameon=False,
                           xticks=[], yticks=[], aspect='equal'))
            super(SkelAxes, self).__init__(fig, rect, *args, **kwargs)
            self.xaxis.set_major_formatter(plt.NullFormatter())
            self.yaxis.set_major_formatter(plt.NullFormatter())
            self.bonearray = pkl.load('bonearray.pkl')
            # define the current rotation
            self.current_rot = Quaternion.from_v_theta((1, 1, 0), np.pi / 6)
        
    
        def draw_bones(self):
            """draw a bone rotated by theta around the given vector"""
            # rotate all the bones
            Rs = [(self.current_rot * rot).as_rotation_matrix() for rot in self.rots]
            bones = [np.dot(self.one_bone, R.T) for R in Rs]
        
            # project the bones - z coordinate for the z-order - ie nothing from behind in front
            bones_proj = [bone[:, :2] for bone in bones]
            zorder = [bone[:4, 2].sum() for bone in bones]
        
            # create the polygons if needed.
            # if they're already drawn, then update them
            if not hasattr(self, '_polys'):
                self._polys = [plt.Polygon(bones_proj[i], fc=self.colors[i],
                                           alpha=0.9, zorder=zorder[i])
                               for i in range()]
                for i in range(6):
                    self.add_patch(self._polys[i])
            else:
                for i in range(6):
                    self._polys[i].set_xy(bones_proj[i])
                    self._polys[i].set_zorder(zorder[i])
                
            self.figure.canvas.draw()
    
