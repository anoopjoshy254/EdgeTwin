import { Routes } from '@angular/router';
import { MachineSelectComponent } from './machine-select/machine-select.component';
import { DashboardComponent } from './dashboard/dashboard.component';

export const routes: Routes = [
    { path: '', component: MachineSelectComponent },
    { path: 'dashboard/:machine', component: DashboardComponent },
    { path: '**', redirectTo: '' }
];
