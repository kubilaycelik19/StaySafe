# Generated by Django 5.1.6 on 2025-04-18 11:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('employees', '0007_alter_employee_imageurl'),
    ]

    operations = [
        migrations.AlterField(
            model_name='employee',
            name='pozisyon',
            field=models.ForeignKey(blank=True, default='isci', null=True, on_delete=django.db.models.deletion.SET_NULL, to='employees.pozisyon'),
        ),
        migrations.AlterField(
            model_name='employee',
            name='vardiya',
            field=models.ForeignKey(blank=True, default='gündüz', null=True, on_delete=django.db.models.deletion.SET_NULL, to='employees.vardiya'),
        ),
    ]
