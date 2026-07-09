import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Renders a high-performance deep-space background starfield using THREE.Points.
 */
export default function Starfield({ count = 3000, radius = 220 }) {
  const [positions, colors, sizes] = useMemo(() => {
    const posArr = new Float32Array(count * 3);
    const colArr = new Float32Array(count * 3);
    const sizeArr = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      // Standard spherical distribution
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);

      // Place stars far away on a sphere shell
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);

      const i3 = i * 3;
      posArr[i3] = x;
      posArr[i3 + 1] = y;
      posArr[i3 + 2] = z;

      // Color variation: white to subtle cold light blue
      const r = 0.85 + Math.random() * 0.15;
      const g = 0.9 + Math.random() * 0.1;
      const b = 0.95 + Math.random() * 0.05;
      colArr[i3] = r;
      colArr[i3 + 1] = g;
      colArr[i3 + 2] = b;

      // Random sizes (0.3px to 1.2px)
      sizeArr[i] = 0.3 + Math.random() * 0.9;
    }

    return [posArr, colArr, sizeArr];
  }, [count, radius]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    return geo;
  }, [positions, colors, sizes]);

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.7}
        sizeAttenuation={true}
        vertexColors={true}
        transparent={true}
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
