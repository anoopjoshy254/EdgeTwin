import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as THREE from 'three';
import gsap from 'gsap';
import { TwinService } from '../twin.service';

interface MachineCard {
    type: string;
    label: string;
    desc: string;
    icon: string;
    color: string;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    renderer?: THREE.WebGLRenderer;
    mesh?: THREE.Mesh;
    animId?: number;
}

@Component({
    selector: 'app-machine-select',
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: './machine-select.component.html',
    styleUrl: './machine-select.component.scss'
})
export class MachineSelectComponent implements AfterViewInit, OnDestroy {
    @ViewChild('latheCanvas') latheCanvasRef!: ElementRef<HTMLCanvasElement>;
    @ViewChild('drillCanvas') drillCanvasRef!: ElementRef<HTMLCanvasElement>;
    @ViewChild('millCanvas') millCanvasRef!: ElementRef<HTMLCanvasElement>;

    showModal = false;
    devices: any[] = [];
    newDevice = { name: '', machine_type: 'Lathe', serial_number: '', location: '' };
    machineOptions = ['Lathe', 'Drill', 'Mill'];

    // Email alert registration
    registeredEmail: string | null = null;
    emailInput = '';
    emailSubmitting = false;
    emailError = '';

    cards: MachineCard[] = [
        { type: 'Lathe', label: 'Lathe', desc: 'Rotational machining with precision chuck control', icon: '⚙️', color: '#00d4ff' },
        { type: 'Drill', label: 'Drill Press', desc: 'High-speed vertical drilling & boring operations', icon: '🔩', color: '#ff6b35' },
        { type: 'Mill', label: 'Milling Machine', desc: 'Multi-axis CNC milling & surface finishing', icon: '🔧', color: '#7c3aed' },
    ];

    constructor(private router: Router, private twin: TwinService) { }

    ngAfterViewInit() {
        // GSAP entrance animation
        gsap.from('.hero-title', { opacity: 0, y: -60, duration: 1, ease: 'power3.out' });
        gsap.from('.hero-sub', { opacity: 0, y: -30, duration: 1, delay: 0.3, ease: 'power3.out' });
        gsap.from('.email-register-bar', { opacity: 0, y: -20, duration: 0.8, delay: 0.5, ease: 'power3.out' });
        gsap.from('.machine-card', {
            opacity: 0, y: 80, stagger: 0.2, duration: 0.9, delay: 0.6, ease: 'back.out(1.7)'
        });
        gsap.from('.register-btn', { opacity: 0, scale: 0.8, duration: 0.6, delay: 1.2, ease: 'back.out(1.7)' });

        const canvases = [this.latheCanvasRef, this.drillCanvasRef, this.millCanvasRef];
        canvases.forEach((ref, i) => this.initThreePreview(ref.nativeElement, this.cards[i]));
        this.twin.getDevices().then(d => this.devices = d);
        // Restore previously registered email from backend
        this.twin.getRegisteredEmail().then(r => { this.registeredEmail = r.email ?? null; });
    }

    async submitEmail() {
        const email = this.emailInput.trim();
        if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            this.emailError = 'Please enter a valid email address.';
            return;
        }
        this.emailError = '';
        this.emailSubmitting = true;
        try {
            const res = await this.twin.registerEmail(email);
            if (res.success) {
                this.registeredEmail = res.email;
                this.emailInput = '';
            } else {
                this.emailError = 'Registration failed. Please try again.';
            }
        } catch {
            this.emailError = 'Cannot connect to server.';
        } finally {
            this.emailSubmitting = false;
        }
    }

    async unregisterEmail() {
        await this.twin.unregisterEmail();
        this.registeredEmail = null;
    }

    initThreePreview(canvas: HTMLCanvasElement, card: MachineCard) {
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setClearColor(0x000000, 0);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
        camera.position.set(0, 0, 4);

        // Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambient);
        const point = new THREE.PointLight(new THREE.Color(card.color), 2.5, 20);
        point.position.set(3, 3, 3);
        scene.add(point);

        // Machine geometry
        let mesh: THREE.Mesh;
        const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(card.color),
            metalness: 0.7,
            roughness: 0.2,
            emissive: new THREE.Color(card.color),
            emissiveIntensity: 0.15
        });

        if (card.type === 'Lathe') {
            const geo = new THREE.CylinderGeometry(0.3, 0.6, 1.8, 32);
            mesh = new THREE.Mesh(geo, mat);
        } else if (card.type === 'Drill') {
            const geo = new THREE.ConeGeometry(0.4, 2, 16);
            mesh = new THREE.Mesh(geo, mat);
        } else {
            const geo = new THREE.BoxGeometry(1.2, 0.5, 1.2);
            mesh = new THREE.Mesh(geo, mat);
        }

        scene.add(mesh);
        card.scene = scene;
        card.camera = camera;
        card.renderer = renderer;
        card.mesh = mesh;

        const animate = () => {
            card.animId = requestAnimationFrame(animate);
            mesh.rotation.y += 0.012;
            mesh.rotation.x += 0.004;
            renderer.render(scene, camera);
        };
        animate();
    }

    hoverCard(card: MachineCard, hover: boolean) {
        if (!card.mesh) return;
        gsap.to(card.mesh.scale, {
            x: hover ? 1.2 : 1, y: hover ? 1.2 : 1, z: hover ? 1.2 : 1,
            duration: 0.4, ease: 'back.out(1.7)'
        });
        (card.mesh.material as THREE.MeshStandardMaterial).emissiveIntensity = hover ? 0.5 : 0.15;
    }

    selectMachine(card: MachineCard) {
        gsap.to('.machine-card', { opacity: 0, y: -40, stagger: 0.1, duration: 0.4, ease: 'power2.in' });
        gsap.to('.hero-title, .hero-sub', { opacity: 0, y: -20, duration: 0.3 });
        setTimeout(() => this.router.navigate(['/dashboard', card.type]), 500);
    }

    openModal() {
        this.showModal = true;
        gsap.from('.modal-box', { scale: 0.7, opacity: 0, duration: 0.4, ease: 'back.out(1.7)' });
    }

    closeModal() {
        gsap.to('.modal-box', {
            scale: 0.85, opacity: 0, duration: 0.25, ease: 'power2.in',
            onComplete: (() => this.showModal = false) as any
        } as any);
    }

    async submitDevice() {
        await this.twin.registerDevice(this.newDevice);
        this.devices = await this.twin.getDevices();
        this.newDevice = { name: '', machine_type: 'Lathe', serial_number: '', location: '' };
    }

    ngOnDestroy() {
        this.cards.forEach(c => {
            if (c.animId) cancelAnimationFrame(c.animId);
            c.renderer?.dispose();
        });
    }
}
