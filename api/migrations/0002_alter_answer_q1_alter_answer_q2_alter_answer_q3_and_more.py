# Generated by Django 4.2.3 on 2023-09-17 03:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='answer',
            name='q1',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='answer',
            name='q2',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='answer',
            name='q3',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='answer',
            name='q4',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='answer',
            name='q5',
            field=models.TextField(blank=True, default=''),
        ),
    ]
