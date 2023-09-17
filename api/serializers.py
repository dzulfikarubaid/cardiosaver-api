from rest_framework import serializers
from .models import User, Answer
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'password', 'c_password']
        extra_kwargs = {
            'password': {
                'write_only': True,
            },
            'c_password': {
                'write_only': True,
            }

        }
    
    def create(self, validated_data):
        password = validated_data.pop('password', None)
        c_password = validated_data.pop('c_password', None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        if c_password is not None:
            instance.set_password(c_password)
        instance.save()
        return instance
        
class AnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Answer
        fields = ('id', 'user', 'q1', 'q2', 'q3', 'q4', 'q5')
