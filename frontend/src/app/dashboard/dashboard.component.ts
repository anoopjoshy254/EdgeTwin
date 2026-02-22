import { Component, OnDestroy, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import * as THREE from 'three';
import gsap from 'gsap';
import { Chart, registerables } from 'chart.js';
import { TwinService, SensorReading } from '../twin.service';

Chart.register(...registerables);

interface ReportCard {
    id: number;
    time: string;
    text: string;
    sensors: { temp: number; vib: number; pwr: number; hum: number };
    failure_count: number;
}

@Component({
    selector: 'app-dashboard',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './dashboard.component.html',
    styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements AfterViewInit, OnDestroy {
    @ViewChild('machineCanvas') machineCanvasRef!: ElementRef<HTMLCanvasElement>;

    machineType = 'Lathe';
    status: 'NORMAL' | 'FAILURE' | 'IDLE' = 'IDLE';
    latest: SensorReading | null = null;
    reports: ReportCard[] = [];
    reportId = 0;

    // Three.js
    private renderer!: THREE.WebGLRenderer;
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private animId!: number;
    private cameraAngle = 0;
    private pt!: THREE.PointLight;
    private _t = 0;

    // Named animated parts (key → mesh for per-machine realism)
    private spinningParts: THREE.Object3D[] = [];  // continuous rotation
    private carriagePart: THREE.Object3D | null = null; // lathe carriage (X slide)
    private carriageOriginX = 0;
    private quillPart: THREE.Object3D | null = null;  // drill quill (Y plunge)
    private quillOriginY = 0;
    private tablePart: THREE.Object3D | null = null;  // mill table (X feed)

    // Failure spark system
    private sparks!: THREE.Points;
    private sparkPos!: Float32Array;
    private sparkVel!: Float32Array;
    private sparksActive = false;

    // Charts
    private charts: { [k: string]: Chart } = {};
    private maxPoints = 40;
    private labels: string[] = [];
    private datasets: { [k: string]: number[] } = { temp: [], vib: [], pwr: [], hum: [] };
    private subs = new Subscription();

    constructor(private route: ActivatedRoute, private router: Router, public twin: TwinService) { }

    ngAfterViewInit() {
        this.machineType = this.route.snapshot.paramMap.get('machine') || 'Lathe';
        this.initThree();
        this.initCharts();

        gsap.from('.dash-header', { opacity: 0, y: -40, duration: 0.7, ease: 'power3.out' });
        gsap.from('.three-panel', { opacity: 0, x: -60, duration: 0.8, delay: 0.2, ease: 'power3.out' });
        gsap.from('.chart-grid', { opacity: 0, y: 40, duration: 0.8, delay: 0.4, ease: 'power3.out' });
        gsap.from('.report-panel', { opacity: 0, x: 60, duration: 0.8, delay: 0.3, ease: 'power3.out' });

        this.twin.connect(this.machineType);
        this.subs.add(this.twin.status$.subscribe(s => { this.status = s; this.onStatusChange(); }));
        this.subs.add(this.twin.reading$.subscribe(r => {
            this.latest = r;
            this.pushToCharts(r);
            if (r.report) this.addReport(r);
        }));
    }

    // ── THREE.JS SCENE ──────────────────────────────────────────────────────────
    initThree() {
        const canvas = this.machineCanvasRef.nativeElement;
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.8;

        this.scene = new THREE.Scene();
        // Rich dark-blue background so machine is visible against it
        this.scene.background = new THREE.Color(0x060e1c);
        // Very light fog — just for depth — NOT density that blacks out the machine
        this.scene.fog = new THREE.FogExp2(0x060e1c, 0.018);

        this.camera = new THREE.PerspectiveCamera(48, canvas.clientWidth / canvas.clientHeight, 0.1, 80);
        this.camera.position.set(6, 3, 6);
        this.camera.lookAt(0, 0, 0);

        // ── Lighting ────────────────────────────────────────────────────
        // Strong ambient so shadows don't go pitch black
        this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
        // Hemisphere sky/ground light
        this.scene.add(new THREE.HemisphereLight(0x88bbff, 0x223344, 1.2));
        // Key light (sun)
        const sun = new THREE.DirectionalLight(0xffffff, 3.5);
        sun.position.set(5, 10, 6);
        sun.castShadow = true;
        sun.shadow.mapSize.set(1024, 1024);
        this.scene.add(sun);
        // Rim light from behind — defines machine silhouette
        const rim = new THREE.DirectionalLight(0x44aaff, 2.0);
        rim.position.set(-6, 4, -6);
        this.scene.add(rim);
        // Warm fill from below
        const fill = new THREE.DirectionalLight(0xffaa44, 0.8);
        fill.position.set(2, -3, 2);
        this.scene.add(fill);
        // Accent point light (machine colour)
        const c = this.machineHex();
        this.pt = new THREE.PointLight(c, 8, 30);
        this.pt.position.set(-1.5, 2.5, 2);
        this.scene.add(this.pt);
        // Second accent from opposite side
        const pt2 = new THREE.PointLight(c, 4, 20);
        pt2.position.set(3, 1, -2);
        this.scene.add(pt2);

        // Build machine
        if (this.machineType === 'Lathe') this.buildLathe();
        else if (this.machineType === 'Drill') this.buildDrill();
        else this.buildMill();

        // Environment — floor & grid
        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(20, 20),
            new THREE.MeshStandardMaterial({ color: 0x08182e, metalness: 0.85, roughness: 0.35 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -1.58;
        floor.receiveShadow = true;
        this.scene.add(floor);
        const grid = new THREE.GridHelper(20, 40, 0x1a5080, 0x0d2840);
        grid.position.y = -1.57;
        this.scene.add(grid);

        this.buildSparks();

        const loop = () => {
            this.animId = requestAnimationFrame(loop);
            this._t += 0.016;
            this.animateMachine();
            this.updateSparks();
            // Slow camera orbit
            this.cameraAngle += 0.0012;
            this.camera.position.x = Math.sin(this.cameraAngle) * 8.5;
            this.camera.position.z = Math.cos(this.cameraAngle) * 8.5;
            this.camera.lookAt(0, 0.2, 0);
            this.renderer.render(this.scene, this.camera);
        };
        loop();
    }

    machineHex(): number {
        return this.machineType === 'Lathe' ? 0x00d4ff
            : this.machineType === 'Drill' ? 0xff6b35 : 0x9b59f5;
    }

    // Base = lighter steel-silver so machine body is visible in scene
    // Accent = vibrant machine colour with strong emissive glow
    m(color: number, metalness = 0.72, roughness = 0.28, noEmissive = false): THREE.MeshStandardMaterial {
        const isBase = noEmissive;
        return new THREE.MeshStandardMaterial({
            color,
            metalness,
            roughness,
            emissive: isBase ? new THREE.Color(0x000000) : new THREE.Color(color),
            emissiveIntensity: isBase ? 0 : 0.55,
        });
    }

    mBase(): THREE.MeshStandardMaterial {
        return new THREE.MeshStandardMaterial({
            color: 0x4a6680,  // visible medium steel blue-grey
            metalness: 0.82,
            roughness: 0.25,
            emissive: new THREE.Color(0x1a2a3a),
            emissiveIntensity: 0.15,
        });
    }

    add(mesh: THREE.Mesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.scene.add(mesh);
        return mesh;
    }

    // ── LATHE ──────────────────────────────────────────────────────────────────
    buildLathe() {
        const c = this.machineHex();
        const accent = this.m(c);

        // Bed (long flat base)
        this.add(mesh(new THREE.BoxGeometry(5.5, 0.32, 1.5), this.mBase(), [0, -1.05, 0]));
        // V-rails
        [-0.48, 0.48].forEach(z =>
            this.add(mesh(new THREE.BoxGeometry(5.5, 0.1, 0.16), this.m(0x5588aa), [0, -0.77, z]))
        );
        // Headstock
        this.add(mesh(new THREE.BoxGeometry(1.25, 1.45, 1.5), this.mBase(), [-1.85, -0.14, 0]));
        // Spindle nose
        const nose = this.add(mesh(new THREE.CylinderGeometry(0.27, 0.27, 0.16, 32), accent, [-1.18, -0.1, 0]));
        nose.rotation.z = Math.PI / 2;
        this.spinningParts.push(nose);
        // 3-jaw chuck (group)
        const chuck = new THREE.Group();
        const chuckDisc = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.44, 0.2, 32), accent);
        chuckDisc.rotation.z = Math.PI / 2;
        chuck.add(chuckDisc);
        for (let i = 0; i < 3; i++) {
            const a = (i / 3) * Math.PI * 2;
            const jaw = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.2, 0.13), this.m(0x99aabb));
            jaw.position.set(0.1, Math.sin(a) * 0.27, Math.cos(a) * 0.27);
            chuck.add(jaw);
        }
        chuck.position.set(-1.03, -0.1, 0);
        this.scene.add(chuck);
        this.spinningParts.push(chuck);
        // Bar stock (workpiece) — shiny silver, no colour emissive
        const bar = this.add(mesh(new THREE.CylinderGeometry(0.13, 0.13, 3.0, 24),
            new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.95, roughness: 0.06 }), [0.65, -0.1, 0]));
        bar.rotation.z = Math.PI / 2;
        this.spinningParts.push(bar);
        // Carriage (slides in X)
        const carriage = this.add(mesh(new THREE.BoxGeometry(0.58, 0.44, 1.28), this.mBase(), [0.5, -0.62, 0]));
        this.carriagePart = carriage;
        this.carriageOriginX = 0.5;
        // Toolpost
        this.add(mesh(new THREE.BoxGeometry(0.2, 0.34, 0.2), this.m(0x5599cc), [0.5, -0.32, -0.36]));
        // Cutting tool — dark steel, no emissive
        this.add(mesh(new THREE.BoxGeometry(0.06, 0.06, 0.46),
            new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 1, roughness: 0.05 }), [0.5, -0.21, -0.1]));
        // Tailstock
        this.add(mesh(new THREE.BoxGeometry(0.68, 0.92, 1.1), this.mBase(), [2.1, -0.56, 0]));
        // Lead screw (spins)
        const screw = this.add(mesh(new THREE.CylinderGeometry(0.033, 0.033, 5.5, 8), this.m(0x5577aa), [0, -0.77, 0.56]));
        screw.rotation.z = Math.PI / 2;
        this.spinningParts.push(screw);
        // Belt cover on headstock
        this.add(mesh(new THREE.BoxGeometry(0.25, 0.8, 1.1), this.mBase(), [-1.85, 0.55, 0]));
    }

    // ── DRILL PRESS ────────────────────────────────────────────────────────────
    buildDrill() {
        const c = this.machineHex();
        const accent = this.m(c);

        // Base plate
        this.add(mesh(new THREE.BoxGeometry(2.2, 0.18, 2.2), this.mBase(), [0, -1.52, 0]));
        // Work table with T-slots
        this.add(mesh(new THREE.BoxGeometry(1.8, 0.12, 1.8), this.mBase(), [0, -0.62, 0]));
        for (let i = -1; i <= 1; i++)
            this.add(mesh(new THREE.BoxGeometry(1.8, 0.06, 0.07),
                new THREE.MeshStandardMaterial({ color: 0x0a1520, metalness: 0.5, roughness: 0.8 }), [0, -0.56, i * 0.36]));
        // Column
        this.add(mesh(new THREE.CylinderGeometry(0.17, 0.22, 4.5, 16), this.mBase(), [-0.62, 0.28, 0]));
        // Motor housing
        this.add(mesh(new THREE.CylinderGeometry(0.31, 0.27, 0.65, 16), this.mBase(), [-0.08, 1.82, 0]));
        // Belt guard
        this.add(mesh(new THREE.BoxGeometry(0.55, 0.5, 0.45), this.mBase(), [-0.35, 1.55, 0]));
        // Head body
        this.add(mesh(new THREE.BoxGeometry(1.4, 0.72, 1.0), this.mBase(), [-0.1, 1.28, 0]));
        // Quill (moves up/down)
        const quill = this.add(mesh(new THREE.CylinderGeometry(0.155, 0.155, 1.15, 16), this.m(0x5599bb), [-0.1, 0.72, 0]));
        this.quillPart = quill;
        this.quillOriginY = 0.72;
        // Spindle (rotates inside quill)
        const spindle = this.add(mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.05, 12), accent, [-0.1, 0.72, 0]));
        this.spinningParts.push(spindle);
        // Drill chuck
        const dc = this.add(mesh(new THREE.CylinderGeometry(0.115, 0.08, 0.22, 16), accent, [-0.1, 0.14, 0]));
        this.spinningParts.push(dc);
        // Drill bit — shiny silver
        const bit = this.add(mesh(new THREE.CylinderGeometry(0.04, 0.004, 0.88, 8),
            new THREE.MeshStandardMaterial({ color: 0xbbbbbb, metalness: 0.97, roughness: 0.06 }), [-0.1, -0.33, 0]));
        this.spinningParts.push(bit);
        // 3 feed handles
        for (let i = 0; i < 3; i++) {
            const a = (i / 3) * Math.PI * 2;
            const arm = this.add(mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.55, 8), this.m(0x4477aa), [-0.1 + Math.cos(a) * 0.27, 0.9, Math.sin(a) * 0.27]));
            arm.rotation.z = Math.PI / 2 + a;
            this.add(mesh(new THREE.SphereGeometry(0.054, 8, 8), this.m(0x224466), [-0.1 + Math.cos(a) * 0.68, 0.9, Math.sin(a) * 0.68]));
        }
        // Laser guide (red line)
        const laser = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 1.5, 4),
            new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0xff0000, emissiveIntensity: 3, transparent: true, opacity: 0.55 }));
        laser.position.set(-0.1, -0.82, 0);
        this.scene.add(laser);
        // Depth stop collar
        this.add(mesh(new THREE.TorusGeometry(0.175, 0.028, 8, 16), this.m(0x888888), [-0.1, 0.56, 0]));
    }

    // ── MILLING MACHINE ────────────────────────────────────────────────────────
    buildMill() {
        const c = this.machineHex();
        const accent = this.m(c);

        // Column base
        this.add(mesh(new THREE.BoxGeometry(3.2, 0.22, 2.2), this.mBase(), [-0.35, -1.52, 0]));
        // Column
        this.add(mesh(new THREE.BoxGeometry(0.6, 3.8, 0.6), this.mBase(), [-1.12, 0.02, 0]));
        // Knee
        this.add(mesh(new THREE.BoxGeometry(1.4, 0.4, 1.55), this.mBase(), [-0.27, -0.57, 0]));
        // Saddle (Y)
        this.add(mesh(new THREE.BoxGeometry(2.4, 0.18, 1.4), this.mBase(), [0, -0.3, 0]));
        // Work table (X feed — oscillates)
        const table = this.add(mesh(new THREE.BoxGeometry(2.9, 0.18, 1.4), this.mBase(), [0, -0.12, 0]));
        this.tablePart = table;
        // T-slots on table
        for (let i = -2; i <= 2; i++)
            this.add(mesh(new THREE.BoxGeometry(2.9, 0.07, 0.07),
                new THREE.MeshStandardMaterial({ color: 0x0a1520, metalness: 0.5, roughness: 0.8 }), [0, -0.04, i * 0.27]));
        // Vise
        this.add(mesh(new THREE.BoxGeometry(0.9, 0.34, 0.72), this.m(0x3a5a7a), [0, 0.08, 0]));
        // Ram
        this.add(mesh(new THREE.BoxGeometry(1.1, 0.34, 0.45), this.mBase(), [-0.45, 1.32, 0]));
        // Spindle head housing
        this.add(mesh(new THREE.CylinderGeometry(0.29, 0.27, 0.65, 20), this.mBase(), [-0.45, 0.83, 0]));
        // Spindle nose
        const sn = this.add(mesh(new THREE.CylinderGeometry(0.12, 0.1, 0.28, 16), accent, [-0.45, 0.45, 0]));
        this.spinningParts.push(sn);
        // Collet / tool holder
        const collet = this.add(mesh(new THREE.CylinderGeometry(0.08, 0.06, 0.2, 12), this.m(0x8899aa), [-0.45, 0.26, 0]));
        this.spinningParts.push(collet);
        // End mill — shiny silver
        const em = this.add(mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.46, 4),
            new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.95, roughness: 0.08 }), [-0.45, -0.01, 0]));
        this.spinningParts.push(em);
        // Chip guard (translucent)
        const g = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 1.2),
            new THREE.MeshStandardMaterial({ color: 0x334466, metalness: 0.3, roughness: 0.5, transparent: true, opacity: 0.22 }));
        g.position.set(0, 0.4, -0.88);
        this.scene.add(g);
    }

    // ── ANIMATION LOOP PER MACHINE ─────────────────────────────────────────────
    animateMachine() {
        const speedMul = this.status === 'FAILURE' ? 2.8 : 1.0;

        if (this.machineType === 'Lathe') {
            // Chuck, nose, bar, screw all spin
            this.spinningParts.forEach(p => { (p as any).rotation.x += 0.055 * speedMul; });
            // Carriage slides back and forth (cutting stroke)
            if (this.carriagePart) {
                this.carriagePart.position.x = this.carriageOriginX + Math.sin(this._t * 0.38) * 0.9;
            }

        } else if (this.machineType === 'Drill') {
            // Spindle, chuck, bit spin around Y
            this.spinningParts.forEach(p => { (p as any).rotation.y += 0.1 * speedMul; });
            // Quill plunges down (drilling stroke) and retracts
            if (this.quillPart) {
                const stroke = Math.max(0, Math.sin(this._t * 0.55)) * 0.55;
                this.quillPart.position.y = this.quillOriginY - stroke;
                // Bit follows quill
                const bit = this.spinningParts[2];
                if (bit) bit.position.y = (this.quillOriginY - stroke) - 1.05;
            }

        } else { // Mill
            // Spindle, collet, end mill spin around Y
            this.spinningParts.forEach(p => { (p as any).rotation.y += 0.13 * speedMul; });
            // Table oscillates in X (XY feed)
            if (this.tablePart) {
                this.tablePart.position.x = Math.sin(this._t * 0.36) * 0.85;
            }
        }

        // Pulse the accent point light
        this.pt.intensity = 4 + Math.sin(this._t * 2.5) * 0.9;
        if (this.status === 'FAILURE') {
            this.pt.color.setHex(0xff1133);
            this.pt.intensity = 7 + Math.sin(this._t * 6) * 2;
        } else {
            this.pt.color.setHex(this.machineHex());
        }
    }

    onStatusChange() {
        if (this.status === 'FAILURE') this.triggerSparks();
        // Flash screen-edge glow handled in CSS with failure-mode class
    }

    // ── SPARK SYSTEM ───────────────────────────────────────────────────────────
    buildSparks() {
        const n = 320;
        this.sparkPos = new Float32Array(n * 3);
        this.sparkVel = new Float32Array(n * 3);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(this.sparkPos, 3));
        const mat = new THREE.PointsMaterial({
            color: 0xff8800, size: 0.08, transparent: true, opacity: 0.95,
            blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
        });
        this.sparks = new THREE.Points(geo, mat);
        this.sparks.visible = false;
        this.scene.add(this.sparks);
    }

    triggerSparks() {
        const n = this.sparkPos.length / 3;
        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            this.sparkPos[i3] = (Math.random() - 0.5) * 0.6;
            this.sparkPos[i3 + 1] = Math.random() * 1.2;
            this.sparkPos[i3 + 2] = (Math.random() - 0.5) * 0.6;
            this.sparkVel[i3] = (Math.random() - 0.5) * 0.12;
            this.sparkVel[i3 + 1] = Math.random() * 0.09 + 0.02;
            this.sparkVel[i3 + 2] = (Math.random() - 0.5) * 0.12;
        }
        this.sparks.geometry.attributes['position'].needsUpdate = true;
        this.sparks.visible = true;
        this.sparksActive = true;
        setTimeout(() => { this.sparks.visible = false; this.sparksActive = false; }, 2800);
    }

    updateSparks() {
        if (!this.sparksActive) return;
        const n = this.sparkPos.length / 3;
        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            this.sparkPos[i3] += this.sparkVel[i3];
            this.sparkPos[i3 + 1] += this.sparkVel[i3 + 1];
            this.sparkPos[i3 + 2] += this.sparkVel[i3 + 2];
            this.sparkVel[i3 + 1] -= 0.0028; // gravity
        }
        this.sparks.geometry.attributes['position'].needsUpdate = true;
    }

    // ── CHARTS ─────────────────────────────────────────────────────────────────
    initCharts() {
        [
            { key: 'temp', label: 'Temperature (°C)', id: 'tempChart', color: '#ff6b35' },
            { key: 'vib', label: 'Vibration (mm/s)', id: 'vibChart', color: '#00d4ff' },
            { key: 'pwr', label: 'Power (kW)', id: 'pwrChart', color: '#7c3aed' },
            { key: 'hum', label: 'Humidity (%)', id: 'humChart', color: '#22c55e' },
        ].forEach(s => {
            const ctx = (document.getElementById(s.id) as HTMLCanvasElement).getContext('2d')!;
            this.charts[s.key] = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: s.label, data: [], borderColor: s.color, backgroundColor: s.color + '18', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true }] },
                options: {
                    responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
                    plugins: { legend: { labels: { color: '#8899aa', font: { family: 'Inter' } } } },
                    scales: {
                        x: { ticks: { color: '#6a7f96', font: { size: 9 } }, grid: { color: '#112233' } },
                        y: { ticks: { color: '#6a7f96', font: { size: 9 } }, grid: { color: '#112233' } }
                    }
                }
            });
        });
    }

    pushToCharts(r: SensorReading) {
        const time = r.time;
        this.labels.push(time);
        if (this.labels.length > this.maxPoints) this.labels.shift();
        (['temp', 'vib', 'pwr', 'hum'] as const).forEach(k => {
            this.datasets[k].push(r[k]);
            if (this.datasets[k].length > this.maxPoints) this.datasets[k].shift();
            if (this.charts[k]) {
                this.charts[k].data.labels = [...this.labels];
                this.charts[k].data.datasets[0].data = [...this.datasets[k]];
                this.charts[k].update('none');
            }
        });
    }

    // ── AI REPORTS ─────────────────────────────────────────────────────────────
    addReport(r: SensorReading) {
        this.reports.unshift({ id: this.reportId++, time: r.time, text: r.report!, sensors: { temp: r.temp, vib: r.vib, pwr: r.pwr, hum: r.hum }, failure_count: r.failure_count });
        if (this.reports.length > 10) this.reports.pop();
        setTimeout(() => {
            const el = document.querySelector('.report-card:first-child');
            if (el) gsap.from(el, { x: 80, opacity: 0, duration: 0.6, ease: 'back.out(1.4)' });
        }, 50);
    }

    goBack() {
        gsap.to('.dashboard-root', { opacity: 0, duration: 0.4, onComplete: (() => this.router.navigate(['/'])) as any } as any);
    }

    ngOnDestroy() {
        this.twin.disconnect();
        this.subs.unsubscribe();
        cancelAnimationFrame(this.animId);
        this.renderer?.dispose();
        Object.values(this.charts).forEach(c => c.destroy());
    }
}

// ── Utility: create a positioned mesh ──────────────────────────────────────────
function mesh(geo: THREE.BufferGeometry, mat: THREE.Material, pos?: [number, number, number]): THREE.Mesh {
    const m = new THREE.Mesh(geo, mat);
    if (pos) m.position.set(...pos);
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
}
