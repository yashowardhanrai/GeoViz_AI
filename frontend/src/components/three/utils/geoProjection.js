import * as THREE from 'three';

/**
 * Converts WGS84 latitude/longitude to 3D Cartesian coordinates
 * aligned with Three.js SphereGeometry UV texture mapping.
 *
 * The formula maps:
 *   x =  R × cos(lat) × cos(lon)
 *   y =  R × sin(lat)
 *   z = -R × cos(lat) × sin(lon)
 *
 * This produces exact alignment with the default SphereGeometry UV wrap
 * so that GeoJSON boundaries sit precisely on top of satellite texture landmasses.
 */
export function latLonToVec3(lat, lon, radius = 1.0) {
  const phi = lat * Math.PI / 180;
  const theta = lon * Math.PI / 180;
  return new THREE.Vector3(
    radius * Math.cos(phi) * Math.cos(theta),
    radius * Math.sin(phi),
    -radius * Math.cos(phi) * Math.sin(theta)
  );
}

/**
 * Fixes antimeridian crossing for polygon rings that span across the
 * 180°/-180° boundary (e.g. Russia's far-east Kamchatka region).
 * Without this fix, Earcut triangulation draws triangles through the
 * globe's interior instead of wrapping around the back.
 */
export function adjustAntimeridian(ring) {
  let hasPos = false;
  let hasNeg = false;
  let minLon = 180;
  let maxLon = -180;

  for (let i = 0; i < ring.length; i++) {
    const lon = ring[i][0];
    if (lon > 0) hasPos = true;
    if (lon < 0) hasNeg = true;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  }

  if (hasPos && hasNeg && (maxLon - minLon > 180)) {
    return ring.map(pt => [pt[0] < 0 ? pt[0] + 360 : pt[0], pt[1]]);
  }
  return ring;
}

/**
 * Converts a 2D GeoJSON polygon coordinate array into a 3D ShapeGeometry
 * conformed to the sphere surface. The shape is first triangulated in 2D
 * (lon/lat plane) using Three.js Earcut, then each vertex is projected
 * onto the sphere using latLonToVec3.
 */
export function geoJsonToSphereGeometry(coords, radius = 1.001) {
  const shape = new THREE.Shape();

  // Outer ring
  const outerRing = coords[0];
  if (!outerRing || outerRing.length === 0) return null;
  shape.moveTo(outerRing[0][0], outerRing[0][1]);
  for (let i = 1; i < outerRing.length; i++) {
    shape.lineTo(outerRing[i][0], outerRing[i][1]);
  }

  // Interior holes
  for (let h = 1; h < coords.length; h++) {
    const holeRing = coords[h];
    if (!holeRing || holeRing.length === 0) continue;
    const holePath = new THREE.Path();
    holePath.moveTo(holeRing[0][0], holeRing[0][1]);
    for (let i = 1; i < holeRing.length; i++) {
      holePath.lineTo(holeRing[i][0], holeRing[i][1]);
    }
    shape.holes.push(holePath);
  }

  // Triangulate in 2D, then project each vertex onto the sphere
  const geom = new THREE.ShapeGeometry(shape);
  const posAttr = geom.getAttribute('position');
  for (let i = 0; i < posAttr.count; i++) {
    const lon = posAttr.getX(i);
    const lat = posAttr.getY(i);
    const vec = latLonToVec3(lat, lon, radius);
    posAttr.setXYZ(i, vec.x, vec.y, vec.z);
  }
  geom.computeVertexNormals();
  return geom;
}

/**
 * Parses a GeoJSON geometry (Polygon or MultiPolygon) into arrays of
 * sphere-conformed mesh geometries and border line point arrays.
 */
export function getCountryGeometries(geometry, radius = 1.001) {
  const geometries = [];
  const borders = [];

  const addPolygon = (coords) => {
    const adjustedCoords = coords.map(ring => adjustAntimeridian(ring));
    const geom = geoJsonToSphereGeometry(adjustedCoords, radius);
    if (geom) geometries.push(geom);

    // Outer boundary ring as 3D line points
    const borderPoints = adjustedCoords[0].map(
      pt => latLonToVec3(pt[1], pt[0], radius + 0.001)
    );
    borders.push(borderPoints);
  };

  if (geometry.type === 'Polygon') {
    addPolygon(geometry.coordinates);
  } else if (geometry.type === 'MultiPolygon') {
    geometry.coordinates.forEach(polyCoords => addPolygon(polyCoords));
  }

  return { geometries, borders };
}
