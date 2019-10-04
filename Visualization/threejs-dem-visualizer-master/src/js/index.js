import * as THREE from "three";
// TODO: OrbitControls import three.js on its own, so the webpack bundle includes three.js twice!
import OrbitControls from "orbit-controls-es6";
import * as Detector from "../js/vendor/Detector";

// TODO: Major performance problems on reading big images
// import * as terrain from "../textures/agri-medium-dem.tif";
// import * as mountainImage from "../textures/agri-medium-autumn.jpg";

import * as terrain from "../textures/agri-small-dem.tif";
import * as mountainImage from "../textures/agri-small-autumn.jpg";
import * as GeoTIFF from "geotiff";

require("../sass/home.sass");

class Application {
  constructor(opts = {}) {
    this.width = window.innerWidth;
    this.height = window.innerHeight

    if (opts.container) {
      this.container = opts.container;
    } else {
      const div = Application.createContainer();
      document.body.appendChild(div);
      this.container = div;
    }

    if (Detector.webgl) {
      this.init();
      this.render();
    } else {
      // TODO: style warning message
      console.log("WebGL NOT supported in your browser!");
      const warning = Detector.getWebGLErrorMessage();
      this.container.appendChild(warning);
    }
  }

  init() {
    this.scene = new THREE.Scene();
    this.setupRenderer();
    this.setupCamera();
    this.setupControls();
    this.setupLight();
    this.setupTerrainModel();
    this.setupHelpers();

    window.addEventListener("resize", () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      this.renderer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    });
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    // when render is invoked via requestAnimationFrame(this.render) there is
    // no 'this', so either we bind it explicitly or use an es6 arrow function.
    // requestAnimationFrame(this.render.bind(this));
    requestAnimationFrame(() => this.render());
  }

  static createContainer() {
    const div = document.createElement("div");
    div.setAttribute("id", "canvas-container");
    div.setAttribute("class", "container");
    // div.setAttribute('width', window.innerWidth);
    // div.setAttribute('height', window.innerHeight);
    return div;
  }

  setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setClearColor(0xd3d3d3); // it's a dark gray
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(this.width, this.height);
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);
  }

  setupCamera() {
    const fov = 75;
    const aspect = this.width / this.height;
    const near = 0.1;
    const far = 10000;
    this.camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this.camera.position.set(1000, 1000, 1000);
    this.camera.lookAt(this.scene.position);
  }

  setupControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enabled = true;
    this.controls.maxDistance = 1500;
    this.controls.minDistance = 0;
    this.controls.autoRotate = true;
  }

  setupLight() {
    this.light = new THREE.DirectionalLight(0xffffff);
    this.light.position.set(500, 1000, 250);
    this.scene.add(this.light);
    // this.scene.add(new THREE.AmbientLight(0xeeeeee));
  }

  setupTerrainModel() {
    const readGeoTif = async () => {
      const rawTiff = await GeoTIFF.fromUrl(terrain);
      const tifImage = await rawTiff.getImage();
      const image = {
        width: tifImage.getWidth(),
        height: tifImage.getHeight()
      };
      

      /* 
      The third and fourth parameter are image segments and we are subtracting one from each,
       otherwise our 3D model goes crazy.
       https://github.com/mrdoob/three.js/blob/master/src/geometries/PlaneGeometry.js#L57
       */
      const geometry = new THREE.PlaneGeometry(
        image.width,
        image.height,
        image.width - 1,
        image.height -1
      );
      const data = await tifImage.readRasters({ interleave: true });

      console.time("parseGeom");
      geometry.vertices.forEach((geom, index) => {
        geom.z = (data[index] / 20) * -1;
      });
      console.timeEnd("parseGeom");

      const texture = new THREE.TextureLoader().load(mountainImage);
      const material = new THREE.MeshLambertMaterial({
        wireframe: false,
        side: THREE.DoubleSide,
        map: texture
      });

      const mountain = new THREE.Mesh(geometry, material);
      mountain.position.y = 0;
      mountain.rotation.x = Math.PI / 2;

      this.scene.add(mountain);

      const loader = document.getElementById("loader");
      loader.style.opacity = "-1";

      // After a proper animation on opacity, hide element to make canvas clickable again
      setTimeout(
        (() => {
          loader.style.display = "none";
        }),
        1500
      );
    };

    readGeoTif();
  }

  setupHelpers() {
    const gridHelper = new THREE.GridHelper(1000, 40);
    this.scene.add(gridHelper);

    // const dirLightHelper = new THREE.DirectionalLightHelper(this.light, 10);
    // this.scene.add(dirLightHelper);

    console.log("The X axis is red. The Y axis is green. The Z axis is blue.");
    const axesHelper = new THREE.AxesHelper(500);
    this.scene.add(axesHelper);
  }
}

// wrap everything inside a function scope and invoke it (IIFE, a.k.a. SEAF)
(() => {
  const app = new Application({
    container: document.getElementById("canvas-container")
  });
  console.log(app);
})();
