import ezdxf
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import math
import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.ops import linemerge, polygonize
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import linemerge, polygonize

def plot_dxf(file_path):
    # Load DXF file
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(f'DXF View: {file_path}')
    
    for entity in msp:
        if entity.dxftype() in ['LINE']:
            start = entity.dxf.start
            end = entity.dxf.end
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')

        if entity.dxftype() == 'LWPOLYLINE':
            # Extract (x, y) coordinates from points
            points = [(point[0], point[1]) for point in entity]
            x, y = zip(*points)
            # Check if polyline is closed
            if entity.is_closed:
                x += (x[0],)
                y += (y[0],)

            ax.plot(x, y, 'b-')
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            circle = plt.Circle(center[:2], radius, fill=False, color='g')
            ax.add_patch(circle)

        elif entity.dxftype() == 'ARC':
            import numpy as np
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            angles = np.linspace(start_angle, end_angle, 100)
            x = center[0] + radius * np.cos(np.radians(angles))
            y = center[1] + radius * np.sin(np.radians(angles))
            ax.plot(x, y, 'r--')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def extract_polygon_from_lines(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    lines = []
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            lines.append(LineString([start, end]))

    if not lines:
        print("No LINE entities found.")
        return []

    # Merge and form closed polygon(s)
    merged = linemerge(lines)
    polygons = list(polygonize(merged))

    if not polygons:
        print("Warning: Lines did not form a closed polygon.")
    return polygons  # List of shapely Polygon(s)

def extract_closed_polylines(dxf_path, circle_res=64):
    """
    Read the DXF you create (CIRCLE + LWPOLYLINE entities) and return
    a list of Shapely polygons representing every closed outline found.

    Parameters
    ----------
    dxf_path : str
        File path of the DXF.
    circle_res : int, optional
        Number of segments used to approximate each CIRCLE (default 64).

    Returns
    -------
    polygons : list[shapely.geometry.Polygon]
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    polygons      = []
    loose_points  = []
    line_segments = []

    # ------------------------------------------------------------------
    #  pass 1 – collect entities
    # ------------------------------------------------------------------
    for ent in msp:
        etype = ent.dxftype()

        # ----- your generated LWPOLYLINEs -----------------------------
        if etype == "LWPOLYLINE":
            # 'closed' flag can be read via .closed or by bit-flag 1
            if getattr(ent, "closed", False) or (ent.dxf.flags & 1):
                pts = [(x, y) for x, y, *_ in ent.get_points("xy")]
                if len(pts) >= 3:
                    polygons.append(Polygon(pts))
            # if not closed, fall through and treat as separate lines

        # ----- possible legacy POLYLINE -------------------------------
        elif etype == "POLYLINE":
            if ent.is_closed:
                verts = [(v.dxf.location.x, v.dxf.location.y) for v in ent.vertices]
                if len(verts) >= 3:
                    polygons.append(Polygon(verts))

        # ----- single CIRCLE (outer bound) ----------------------------
        elif etype == "CIRCLE":
            centre = (ent.dxf.center.x, ent.dxf.center.y)
            radius = ent.dxf.radius
            circle_poly = Point(centre).buffer(radius, resolution=circle_res)
            polygons.append(circle_poly)

        # ----- any HATCH boundary ------------------------------------
        elif etype == "HATCH":
            for path in ent.paths:
                if path.path_type_flags == 1:          # external loop
                    pts = [(v[0], v[1]) for v in path.edges if hasattr(v, "__getitem__")]
                    if len(pts) >= 3:
                        polygons.append(Polygon(pts))

        # ----- stand-alone LINE entities ------------------------------
        elif etype == "LINE":
            p0 = (ent.dxf.start.x, ent.dxf.start.y)
            p1 = (ent.dxf.end.x,  ent.dxf.end.y)
            line_segments.append(LineString((p0, p1)))

        # ----- left-over POINTs (rare) -------------------------------
        elif etype == "POINT":
            loose_points.append((ent.dxf.location.x, ent.dxf.location.y))

    # ------------------------------------------------------------------
    #  pass 2 – try to close any stray LINE network into polygons
    # ------------------------------------------------------------------
    # if line_segments:
    #     merged   = linemerge(line_segments)
    #     polylist = list(polygonize(merged))
    #     if polylist:
    #         polygons.extend(polylist)
    #     else:
    #         print("⚠  LINE entities did not form a closed loop.")

    # ------------------------------------------------------------------
    #  pass 3 – last-resort: try to build polygon from scattered POINTs
    # ------------------------------------------------------------------
    if len(loose_points) >= 3:
        cand = Polygon(loose_points)
        if cand.is_valid:
            polygons.append(cand)
        else:
            print("⚠  POINT cloud forms an invalid polygon.  Ignored.")

    return polygons

def compute_total_area(polygons, units_to_meters=1.0):
    union = unary_union(polygons)
    return union.area * (units_to_meters ** 2)





####################################################################################





def intersection_area(shape1, shape2):
    
    poly1 = shape1 
    poly2 = shape2  
    if type(poly1) is  list:
        poly1 = poly1[0] 
    else: 
        pass

    if type(poly2) is  list:
            poly2 = poly2[0] 
    else: 
        pass


    
    intersection = poly1.intersection(poly2)
    # print(f"Intersection area (in m²): {intersection.area * (units ** 2)}")
    area = intersection.area * (units ** 2)
    return area



def rotate_shape(shape, angle, centroid=(0, 0)):
    return shapely_rotate(shape[0], angle=angle, origin=centroid)

def rotation_change_area(shape1, shape2, angles , step=10 , plot = False):
    area_change = []
    rot_shape =  shape1 

    for theta in angles:
        rot_shape = rotate_shape(shape1, theta, centroid=(0, 0))
        area = intersection_area(rot_shape, shape2)
        area_change.append(area)

        rot_shape = [rot_shape]

        # --- Plot each rotated shape with intersection overlay ---

        if plot == True: 
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

            # Original shape 2 (fixed)
            x2, y2 = shape2[0].exterior.xy
            ax.plot(x2, y2, 'r-', label='Shape 2 (fixed)')

            # Rotated shape
            x1, y1 = rot_shape[0].exterior.xy
            ax.plot(x1, y1, 'b-', label=f'Rotated Shape (θ={theta}°)')

            # Intersection overlay
            inter = shape2[0].intersection(rot_shape)[0]
            if not inter.is_empty and inter.geom_type == 'Polygon':
                xi, yi = inter.exterior.xy
                ax.fill(xi, yi, color='green', alpha=0.3, label='Intersection')

            ax.legend()
            plt.title(f"Rotation θ = {theta}° | Overlap Area = {area:.6f}")
            plt.grid(True)
            plt.show()
            # Optional: pause or save figure here

    return area_change


from shapely.geometry import LineString, Polygon, MultiLineString

def lineintegrals_sum(shape: Polygon,
                      step_deg: float = 1.0,
                      overshoot: float = 1.2) -> np.ndarray:
    """
    Compute the length of the intersection between a rotating
    diameter-line and a polygon for every angle.

    Parameters
    ----------
    shape : shapely.geometry.Polygon
        The plate outline, assumed to be centred on (0, 0).
    step_deg : float, optional
        Angular increment Δθ in degrees; default 1 deg → 360 samples.
    overshoot : float, optional
        Factor by which the probing line is longer than the plate’s
        largest diagonal, so it is guaranteed to cross the polygon.

    Returns
    -------
    lengths : numpy.ndarray
        Array of chord lengths, one per sampled angle θ = 0, Δθ, 2Δθ, …
    """

    # --- choose a line long enough to cover the whole polygon -------------
    minx, miny, maxx, maxy = shape[0].bounds
    half_diag = 0.5 * np.hypot(maxx - minx, maxy - miny)
    L = overshoot * half_diag              # half-length of the test line

    angles_deg = np.arange(0, 360, step_deg)
    lengths = np.zeros_like(angles_deg, dtype=float)

    for idx, θ in enumerate(angles_deg):
        theta = np.deg2rad(θ)
        dx, dy = np.cos(theta), np.sin(theta)

        # build the diameter-line centred at origin, length 2 L
        p0 = (-L*dx, -L*dy)
        p1 = ( L*dx,  L*dy)
        probe_line = LineString([p0, p1])

        inter = shape[0].intersection(probe_line)

        # intersection can be LineString, MultiLineString, Point, or empty
        if inter.is_empty:
            lengths[idx] = 0.0
        elif isinstance(inter, LineString):
            lengths[idx] = inter.length
        elif isinstance(inter, MultiLineString):
            lengths[idx] = sum(seg.length for seg in inter.geoms)
        else:  # Point or MultiPoint → zero length along the line
            lengths[idx] = 0.0

    return lengths
    


# ---- Example usage ----
if __name__ == "__main__":
    dxf_file_1 = "/Users/michael_adlerstein/Desktop/kings stuff /Rheology_code/gt1.dxf"  # Replace with your DXF filename
    dxf_file_2 = "/Users/michael_adlerstein/Desktop/kings stuff /Rheology_code/val.dxf"  # Replace with your DXF filename

    plot_dxf(dxf_file_1) 
    plot_dxf(dxf_file_2) 


    units = 0.001  # Set to 0.001 if drawing is in mm; 1.0 if already in meters

    shape_1 = extract_closed_polylines(dxf_file_1)
    shape_2 = extract_closed_polylines(dxf_file_2)
    # print(shape_1)

    area_m1 = compute_total_area(shape_1, units_to_meters=units)
    area_m2 = compute_total_area(shape_2, units_to_meters=units)
    print(f"Total area shape 1 : {area_m1:.6f} m²")
    print(f"Total area shape 2 : {area_m2:.6f} m²")

    intersection_area(shape_1 , shape_2)

    rot_shape_1 = rotate_shape(shape_1 , 10 , centroid = (0,0))
    area_rot = compute_total_area(rot_shape_1, units_to_meters=units)

    print(f"Total area shape 2 : {area_rot} m²")

    step = 1
    theta = range(0, 360, step)
    area = rotation_change_area(shape_1 , shape_2 ,theta , step = step , plot=False )
    print(area)
    plt.plot(theta , area)
    theta_rad = np.radians(theta)

    # Plot in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_rad, area)

    ax.set_title("Intersection Area vs Rotation Angle (Polar Plot)", va='bottom')
    plt.show()
    plt.show()

    area = np.array(area)
    fft_vals = np.fft.fft(area)
    fft_freqs = np.fft.fftfreq(len(area), d=step)  # step = 1 degree

    # Keep only positive frequencies
    positive_freqs = fft_freqs[:len(fft_freqs)//2]
    positive_magnitudes = np.abs(fft_vals[:len(fft_vals)//2])

    # Plot frequency spectrum
    plt.figure()
    plt.plot(positive_freqs, positive_magnitudes)
    plt.xlabel("Frequency (cycles per 360°)")
    plt.ylabel("Amplitude")
    plt.title("FFT of Area vs Rotation Angle")
    plt.grid(True)
    plt.show()


    chords = lineintegrals_sum(shape_1, step_deg=1.0)   # 360 line-integrals
    theta = np.deg2rad(np.arange(0, 360, 1.0))
    # plt.subplot(projection='polar')
    plt.plot(theta, chords)
    plt.title("Chord length vs direction")
    plt.show()